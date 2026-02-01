import torch
import torch.nn as nn
import math

class DINOv3RoPE(nn.Module):
    """
    模拟 DINOv3 的旋转位置编码
    """
    def __init__(self, dim, max_len=2048):
        super().__init__()
        self.dim = dim
        self.max_len = max_len
        
        # 生成频率矩阵 (与官方实现一致)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_len).float()
        freqs = torch.outer(t, inv_freq)
        
        # 注册为 buffer，不参与更新
        self.register_buffer("cos", freqs.cos())
        self.register_buffer("sin", freqs.sin())

    def forward(self, q, k):
        # q, k shape: [B, H, N, D]
        # 简单的 RoPE 应用逻辑
        B, H, N, D = q.shape
        cos = self.cos[:N, :].unsqueeze(0).unsqueeze(0)  # [1, 1, N, D/2]
        sin = self.sin[:N, :].unsqueeze(0).unsqueeze(0)
        
        # 将最后一维 split 为 2 部分进行旋转
        q_r1, q_r2 = q.chunk(2, dim=-1)
        k_r1, k_r2 = k.chunk(2, dim=-1)
        
        # 执行旋转: [x, y] -> [x*cos - y*sin, x*sin + y*cos]
        # 注意：这里会再次推高数值
        q_new = torch.cat([q_r1 * cos - q_r2 * sin, q_r1 * sin + q_r2 * cos], dim=-1)
        k_new = torch.cat([k_r1 * cos - k_r2 * sin, k_r1 * sin + k_r2 * cos], dim=-1)
        
        return q_new, k_new

class SingleLayerTransformer(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # 线性层
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.rope = DINOv3RoPE(self.head_dim)

    def forward(self, x):
        B, N, C = x.shape
        
        # 1. QKV 投影
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] # [B, H, N, D]
        
        print(f"  [Probe] QKV Projection Max: {q.abs().max().item():.2f}")

        # 2. 应用 RoPE
        q_rope, k_rope = self.rope(q, k)
        print(f"  [Probe] After RoPE Max:     {q_rope.abs().max().item():.2f}")

        # 3. 计算 Attention Score (危险区域!)
        # attn = (q @ k.T) * scale
        attn_scores = (q_rope @ k_rope.transpose(-2, -1))
        
        # --- 关键观测点 ---
        max_score = attn_scores.abs().max().item()
        is_inf = torch.isinf(attn_scores).any().item()
        is_nan = torch.isnan(attn_scores).any().item()
        
        status = "❌ INF/NAN DETECTED" if (is_inf or is_nan) else "✅ OK"
        print(f"  [Probe] Pre-Softmax Dot Max: {max_score:.2f}")
        print(f"  [Probe] Calculation Status:  {status}")
        
        attn_scores = attn_scores * self.scale
        attn_probs = attn_scores.softmax(dim=-1)
        
        return attn_probs

class SingleLayerProtectTransformer(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False) 
        self.rope = DINOv3RoPE(self.head_dim)
        self.scale = self.head_dim ** -0.5

    def forward(self, x, protection_mode="none"):
        """
        protection_mode: 
          - "none": 不保护，直接硬刚 (FP16下必挂)
          - "fp32_cast": 强制转 FP32 计算 (混合精度，最稳)
          - "scaling": 缩放输入，计算完再还原 (纯 FP16 技巧)
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # RoPE 后的值可能很大 (~60000)
        q_rope, k_rope = self.rope(q, k)

        # ================== 核心保护区 ==================
        if protection_mode == "fp32_cast":
            # 🛡️ 策略 A: 混合精度 (TensorRT 的 --layerOutputTypes 对应此法)
            # 临时将输入提升为 FP32，计算完矩阵乘法后再看情况转回或保持
            # FP32 上限是 10^38，完全能吃下 39 亿
            with torch.cuda.amp.autocast(enabled=False):
                q_safe = q_rope.float()
                k_safe = k_rope.float()
                attn_scores = (q_safe @ k_safe.transpose(-2, -1))
                # 注意：此时 attn_scores 是 FP32 的，数值是正确的 39亿
        
        elif protection_mode == "scaling":
            # 🛡️ 策略 B: 数学缩放 (Pre-Scaling)
            # 如果强制要求全程 FP16，就把输入缩小，让积变小，最后再补偿回来
            # 60000 / 32 ≈ 1875。 1875^2 * 64 ≈ 2.2亿 (还是有点大，更安全是除以 64)
            scale_factor = 64.0 
            q_safe = q_rope / scale_factor
            k_safe = k_rope / scale_factor
            
            # 计算点积 (此时结果只有几百万，FP16 安全)
            attn_scores = (q_safe @ k_safe.transpose(-2, -1))
            
            # 补偿缩放: 结果 = 原结果 / (scale * scale)
            # 所以要乘回 scale^2
            attn_scores = attn_scores * (scale_factor ** 2)
            
        else:
            # 💀 无保护 (裸奔)
            attn_scores = (q_rope @ k_rope.transpose(-2, -1))
        # ===============================================

        # 检查状态
        max_score = attn_scores.abs().max().item()
        is_inf = torch.isinf(attn_scores).any().item()
        status = "❌ INF" if is_inf else "✅ OK"
        
        print(f"  [Mode: {protection_mode:<10}] Max: {max_score:.2E} | Status: {status}")
        return attn_scores

def run_verification():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 模拟配置
    DIM = 384
    HEADS = 6
    SEQ_LEN = 1024 # 32x32 patches
    
    model = SingleLayerProtectTransformer(DIM, HEADS).to(device)
    model.eval()
    
    # --- 构造高能输入 ---
    # 我们需要模拟 DINOv3 在网络深层时的特征强度
    # 经过 RoPE 后达到 63000，说明输入 magnitude 大约在 63000 / sqrt(2) 左右 (粗略估计)
    # 或者直接输入较大的随机数
    input_tensor = torch.randn(1, SEQ_LEN, DIM).to(device) * 2200.0  # 放大输入信号
    
    print("="*60)
    print("🧪 实验 1: FP32 模式 (基准)")
    print("="*60)
    with torch.no_grad():
        model(input_tensor)

    print("\n" + "="*60)
    print("🧪 实验 2: FP16 模式 (模拟 TensorRT 半精度)")
    print("="*60)
    
    # 强制将模型和输入转为 FP16，模拟 TensorRT 的行为
    model_half = model.half()
    input_half = input_tensor.half()
    
    with torch.no_grad():
        try:
            model_half(input_half, protection_mode="none")
            model_half(input_half, protection_mode="fp32_cast")
            model_half(input_half, protection_mode="scaling")
        except RuntimeError as e:
            print(f"Runtime Error: {e}")

if __name__ == "__main__":
    run_verification()
