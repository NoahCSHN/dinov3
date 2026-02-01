import torch
import torch.nn as nn
import os
import warnings
import onnx
from onnxsim import simplify

# 屏蔽无关的 Trace 警告
warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)

class DINOv3ConvNeXtWrapper(nn.Module):
    def __init__(self, local_repo_dir, model_name, weights_path):
        super().__init__()
        # 加载 DINOv3 的 ConvNeXt-Tiny 版本
        self.model = torch.hub.load(
            local_repo_dir, 
            model_name, 
            source='local', 
            pretrained=False  # 先不加载默认权重，手动加载
        )
        
        # 手动加载权重 (更稳健)
        if os.path.exists(weights_path):
            print(f"[*] Loading weights from: {weights_path}")
            checkpoint = torch.load(weights_path, map_location='cpu')
            # 兼容处理：有时候权重在 'model' 或 'teacher' 键下
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'teacher' in checkpoint:
                state_dict = checkpoint['teacher']
            else:
                state_dict = checkpoint
                
            # 移除 DDP 前缀 (如果有)
            state_dict = {k.replace("module.", "").replace("backbone.", ""): v for k, v in state_dict.items()}
            
            # 加载权重 (strict=False 以容忍分类头缺失)
            msg = self.model.load_state_dict(state_dict, strict=False)
            print(f"[*] Weights loaded. Missing keys (expected for backbone): {len(msg.missing_keys)}")
        else:
            print(f"❌ Weights file not found: {weights_path}")
            
        self.model.float()

    def forward(self, x):
        # DINOv3 ConvNeXt 的特征提取方法可能略有不同
        # 通常是用 forward_features 或 get_intermediate_layers
        # ConvNeXt 输出通常是 [B, C, H, W]
        features = self.model.forward_features(x)
        return features

def export_convnext_onnx():
    # --- 配置区域 ---
    local_repo_dir = '/home/wayrobo/0_code/dinov3'  # DINOv3 源码路径
    
    # 这里的名称取决于 dinov3/hubconf.py 里的定义
    # 常见的可能是: 'dinov3_convnext_tiny', 'dinov3_convnext_small' 等
    model_name = "dinov3_convnext_tiny" 
    
    # 请确保这是 ConvNeXt 版本的权重文件 (.pth)
    weights_path = '/home/wayrobo/0_code/dinov3/pretrained/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth' 
    
    output_onnx = "dinov3_convnext_tiny_512.onnx"
    input_size = (1, 3, 512, 512)
    # ----------------
    
    device = "cpu"
    print(f"[*] Initializing model: {model_name}...")
    
    try:
        wrapper = DINOv3ConvNeXtWrapper(local_repo_dir, model_name, weights_path).to(device)
        wrapper.eval()
    except Exception as e:
        print(f"❌ Model initialization failed: {e}")
        print("💡 Hint: Check if 'dinov3_convnext_tiny' is defined in hubconf.py")
        return

    # 创建 Dummy Input
    dummy_input = torch.randn(*input_size, device=device)

    print(f"[*] Starting ONNX export to {output_onnx}...")
    
    # 导出
    torch.onnx.export(
        wrapper,
        dummy_input,
        output_onnx,
        export_params=True,
        opset_version=18,  # 推荐 17 以支持较新的算子
        do_constant_folding=True,
        input_names=['input'],
        output_names=['feature_map'], # ConvNeXt 输出是特征图
        dynamic_axes={
            'input': {0: 'batch_size', 2: 'height', 3: 'width'},
            'feature_map': {0: 'batch_size', 2: 'height', 3: 'width'}
        }
    )
    print(f"✅ Raw ONNX exported.")

    # 简化 (onnx-sim)
    print(f"[*] Simplifying model...")
    try:
        model_onnx = onnx.load(output_onnx)
        model_simp, check = simplify(model_onnx)
        if check:
            onnx.save(model_simp, output_onnx)
            print(f"✅ Simplified ONNX saved to {output_onnx}")
        else:
            print("⚠️ Simplification check failed, saved raw model.")
    except Exception as e:
        print(f"⚠️ onnx-sim failed: {e}")

if __name__ == "__main__":
    export_convnext_onnx()
