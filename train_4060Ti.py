import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm
from einops import rearrange, repeat
import platform


# 开启 TF32 (Ampere/Ada 架构核心加速)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# 导入新版 autocast
try:
    from torch.amp import autocast
except ImportError:
    from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

# ==========================================
# 1. 配置 (Ada Lovelace Config)
# ==========================================
class Config:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    img_size = 32
    patch_size = 2
    in_channels = 3
    num_classes = 10
    
    # Modern ViT 架构参数
    hidden_size = 384       
    depth = 8               
    num_heads = 6
    mlp_ratio = 4.0
    bottleneck_dim = 64     
    
    in_context_len = 8      
    in_context_start = 2    
    
    # --- 针对 4060 Ti 的显存优化 ---
    # 4060 Ti 8G 建议: 64
    # 4060 Ti 16G 建议: 128
    batch_size = 64       
    
    # 梯度
    gradient_accumulation_steps = 4
    
    lr = 5e-4
    epochs = 200
    

    num_workers = 0 if platform.system() == 'Windows' else 4
    
    noise_scale = 1.0       
    save_dir = "./jit_results_4060ti"
    resume_path = None

# ==========================================
# 2. 基础组件
# # ==========================================
def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
    def forward(self, x):
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.variance_epsilon)
        return self.weight * x

class VisionRotaryEmbeddingFast(nn.Module):
    def __init__(self, dim, pt_seq_len, num_cls_token=0):
        super().__init__()
        self.dim = dim
        self.pt_seq_len = pt_seq_len 
        freqs = 1. / (10000 ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        t = torch.arange(pt_seq_len)
        freqs = torch.einsum('i,j->ij', t, freqs) 
        freqs = torch.cat((freqs[:, None, :].repeat(1, pt_seq_len, 1), 
                           freqs[None, :, :].repeat(pt_seq_len, 1, 1)), dim=-1) 
        freqs = rearrange(freqs, 'h w d -> (h w) d') 
        freqs = repeat(freqs, 'n d -> n (d r)', r=2) 
        if num_cls_token > 0:
            pad = torch.zeros(num_cls_token, freqs.shape[1])
            freqs = torch.cat([pad, freqs], dim=0)
        self.register_buffer("freqs_cos", freqs.cos())
        self.register_buffer("freqs_sin", freqs.sin())

    def forward(self, x):
        x1, x2 = x[..., 0::2], x[..., 1::2]
        x_rotated = torch.cat((-x2, x1), dim=-1)
        seq_len = x.shape[-2]
        f_cos = self.freqs_cos[:seq_len]
        f_sin = self.freqs_sin[:seq_len]
        return x * f_cos + x_rotated * f_sin

class SwiGLUFFN(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        hidden_dim = int(hidden_dim * 2 / 3)
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(dim, hidden_dim)
        self.w3 = nn.Linear(hidden_dim, dim)
    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class Attention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
    def forward(self, x, rope):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = self.q_norm(q), self.k_norm(k)
        q, k = rope(q), rope(k)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj(x)

class JiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.attn = Attention(hidden_size, num_heads)
        self.norm2 = RMSNorm(hidden_size)
        self.mlp = SwiGLUFFN(hidden_size, int(hidden_size * mlp_ratio))
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
    def forward(self, x, c, rope_func):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), rope=rope_func)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class BottleneckPatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim, bottleneck_dim):
        super().__init__()
        self.proj1 = nn.Conv2d(in_chans, bottleneck_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.proj2 = nn.Conv2d(bottleneck_dim, embed_dim, kernel_size=1, stride=1, bias=True)
    def forward(self, x):
        return self.proj2(self.proj1(x)).flatten(2).transpose(1, 2).contiguous()

# ==========================================
# 3. JiT 主模型
# ==========================================
class JiT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.in_channels = config.in_channels
        self.out_channels = config.in_channels
        self.hidden_size = config.hidden_size
        self.t_embedder = nn.Sequential(nn.Linear(256, config.hidden_size), nn.SiLU(), nn.Linear(config.hidden_size, config.hidden_size))
        self.y_embedder = nn.Embedding(config.num_classes + 1, config.hidden_size)
        self.x_embedder = BottleneckPatchEmbed(config.img_size, config.patch_size, config.in_channels, config.hidden_size, config.bottleneck_dim)
        hw_seq_len = config.img_size // config.patch_size
        head_dim = config.hidden_size // config.num_heads
        self.feat_rope = VisionRotaryEmbeddingFast(dim=head_dim // 2, pt_seq_len=hw_seq_len, num_cls_token=0)
        self.feat_rope_incontext = VisionRotaryEmbeddingFast(dim=head_dim // 2, pt_seq_len=hw_seq_len, num_cls_token=config.in_context_len)
        self.blocks = nn.ModuleList([JiTBlock(config.hidden_size, config.num_heads, config.mlp_ratio) for _ in range(config.depth)])
        self.norm_final = RMSNorm(config.hidden_size)
        self.linear_final = nn.Linear(config.hidden_size, config.patch_size**2 * self.out_channels)
        self.adaLN_final = nn.Sequential(nn.SiLU(), nn.Linear(config.hidden_size, 2 * config.hidden_size))

    def get_time_embedding(self, t):
        half_dim = 256 // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(start=0, end=half_dim, dtype=torch.float32) / half_dim).to(t.device)
        args = t[:, None].float() * freqs[None]
        return self.t_embedder(torch.cat([torch.cos(args), torch.sin(args)], dim=-1))

    def unpatchify(self, x):
        p = self.config.patch_size
        c = self.out_channels
        h = w = int(x.shape[1] ** 0.5)
        return torch.einsum('nhwpqc->nchpwq', x.reshape(shape=(x.shape[0], h, w, p, p, c))).reshape(shape=(x.shape[0], c, h * p, h * p))

    def forward(self, x, t, y):
        c = self.get_time_embedding(t) + self.y_embedder(y)
        x = self.x_embedder(x)
        for i, block in enumerate(self.blocks):
            if i == self.config.in_context_start:
                ctx = self.y_embedder(y).unsqueeze(1).repeat(1, self.config.in_context_len, 1)
                x = torch.cat([ctx, x], dim=1)
            x = block(x, c, self.feat_rope if i < self.config.in_context_start else self.feat_rope_incontext)
        if self.config.in_context_len > 0: x = x[:, self.config.in_context_len:]
        shift, scale = self.adaLN_final(c).chunk(2, dim=1)
        return self.unpatchify(self.linear_final(modulate(self.norm_final(x), shift, scale)))

# ==========================================
# 4. 扩散逻辑 (Flow Matching)
# ==========================================
class Diffusion:
    def __init__(self, config):
        self.noise_scale = config.noise_scale
        self.device = config.device
    def sample_t(self, n):
        return torch.sigmoid(torch.randn(n, device=self.device) * 1.2 - 0.6)
    def q_sample(self, x0, t):
        t = t.view(-1, 1, 1, 1)
        e = torch.randn_like(x0) * self.noise_scale
        z = t * x0 + (1 - t) * e
        return z, x0 - e, e 
    
    @torch.no_grad()
    def p_sample_loop(self, model, n, labels, steps=50):
        model.eval()
        z = torch.randn(n, 3, Config.img_size, Config.img_size, device=self.device) * self.noise_scale
        dt = 1.0 / steps
        null_labels = torch.full_like(labels, Config.num_classes)
        for i in range(steps):
            t_curr = torch.tensor(i / steps, device=self.device).repeat(n)
            z_in = torch.cat([z, z])
            t_in = torch.cat([t_curr, t_curr])
            y_in = torch.cat([labels, null_labels])
            x_pred = model(z_in, t_in, y_in)
            denom = (1 - t_in.view(-1, 1, 1, 1)).clamp(min=1e-3)
            v_pred_both = (x_pred - z_in) / denom
            v_cond, v_uncond = v_pred_both.chunk(2)
            v = v_uncond + 3.0 * (v_cond - v_uncond)
            z = z + v * dt
        model.train()
        return z.clamp(-1, 1)

# ==========================================
# 5. 主程序 
# ==========================================
def main():
    os.makedirs(Config.save_dir, exist_ok=True)
    print(f"🔥 RTX 4060 Ti Optimizations | Device: {Config.device}")
    
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = datasets.CIFAR10(root="data", train=True, download=True, transform=transform)
    

    dataloader = DataLoader(
        dataset, 
        batch_size=Config.batch_size, 
        shuffle=True, 
        num_workers=Config.num_workers, 
        pin_memory=True,
        persistent_workers=(Config.num_workers > 0)
    )
            
    model = JiT(Config).to(Config.device)
    
    # Windows需先安装triton
    print("正在编译模型 (torch.compile)...")
    try:
        # mode='reduce-overhead' 会消耗更多显存，这里用默认模式或 max-autotune-no-cudagraphs
        model = torch.compile(model) 
    except Exception as e:
        print(f"⚠️ 编译失败或不支持，将使用常规模式运行: {e}")
    
    optimizer = optim.AdamW(model.parameters(), lr=Config.lr)
    diffusion = Diffusion(Config)
    scaler = torch.amp.GradScaler('cuda')
    
    print("🚀 开始训练...")
    global_step = 0
    
    for epoch in range(Config.epochs):
        pbar = tqdm(dataloader)
        loss_ema = None
        model.train()
        
        optimizer.zero_grad()
        for step, (x, y) in enumerate(pbar):
            x, y = x.to(Config.device, non_blocking=True), y.to(Config.device, non_blocking=True)
            t = diffusion.sample_t(x.shape[0])
            z, v_target, _ = diffusion.q_sample(x, t)
            
            if torch.rand(1) < 0.1:
                y = torch.full_like(y, Config.num_classes)
            
            with autocast('cuda', dtype=torch.bfloat16):
                x_pred = model(z, t, y)
                denom = (1 - t.view(-1, 1, 1, 1)).clamp(min=1e-3)
                v_pred = (x_pred - z) / denom
                loss = F.mse_loss(v_pred, v_target)
                loss = loss / Config.gradient_accumulation_steps
            
            # Backward
            scaler.scale(loss).backward()
            
            # 梯度累积
            if (step + 1) % Config.gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                global_step += 1
            
            # 还原 Loss 用于显示
            current_loss = loss.item() * Config.gradient_accumulation_steps
            if loss_ema is None: loss_ema = current_loss
            else: loss_ema = 0.95 * loss_ema + 0.05 * current_loss
            pbar.set_description(f"E{epoch} | L:{loss_ema:.4f}")

        # 采样测试
        if epoch % 5 == 0:
            print("🎨 正在生成采样图...")
            sample_labels = torch.tensor([7]*8 + [1]*8).to(Config.device) # 8个马，8个车
            with torch.no_grad():
                imgs = diffusion.p_sample_loop(model, 16, sample_labels)
            save_image((imgs + 1)/2, f"{Config.save_dir}/{epoch}.png", nrow=4)
            torch.save(model.state_dict(), f"{Config.save_dir}/last.pth")

if __name__ == "__main__":
    main()