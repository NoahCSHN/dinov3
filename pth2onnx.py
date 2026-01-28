import torch
import torch.nn as nn
import os

# --- 1. 定义 ONNX 导出包装类 ---
class DINOv3ExportWrapper(nn.Module):
    def __init__(self, local_repo_dir, model_name, weights_path):
        super().__init__()
        # 加载基础模型
        self.base_model = torch.hub.load(
            local_repo_dir, 
            model_name, 
            source='local', 
            weights=weights_path
        )
        self.base_model.eval()

    def forward(self, x):
        # 对应你脚本中的 get_intermediate_layers(n=1) 逻辑
        # 在 DINOv3 中，这通常提取最后一个 Transformer Block 的输出
        # 我们显式调用内部方法以保证导出路径清晰
        features = self.base_model.get_intermediate_layers(x, n=1)[0]
        return features

# --- 2. 执行转换 ---
def export_to_onnx():
    # 配置路径（完全匹配你的 pipline_segment.py）
    local_repo_dir = '/home/wayrobo/0_code/dinov3'
    model_name = "dinov3_vits16"
    weights_path = '/home/wayrobo/0_code/dinov3/pretrained/dinov3_vits16_pretrain_lvd1689m-08c60483.pth'
    output_onnx = "dinov3_vits16_512.onnx"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[*] Loading model for export...")
    wrapper = DINOv3ExportWrapper(local_repo_dir, model_name, weights_path).to(device)
    wrapper.eval()

    # 构造静态输入 (1, 3, 512, 512)
    # TensorRT 在静态输入下性能最优，完全满足 20Hz 需求
    dummy_input = torch.randn(1, 3, 512, 512).to(device)

    print(f"[*] Exporting to ONNX (Opset 17)...")
    torch.onnx.export(
        wrapper,
        dummy_input,
        output_onnx,
        export_params=True,
        opset_version=18,        # 推荐 17，能更好地融合 ViT 的 Layernorm 和 Attention
        do_constant_folding=True,
        input_names=['input'],
        output_names=['feature_tokens'],
        # 保持静态 Shape 以获得 Orin 平台的极致加速
        dynamic_axes=None 
    )

    print(f"✅ Export Complete: {output_onnx}")
    
    # 验证导出文件
    if os.path.exists(output_onnx):
        import onnx
        onnx_model = onnx.load(output_onnx)
        onnx.checker.check_model(onnx_model)
        print("✅ ONNX structure verified.")

if __name__ == "__main__":
    export_to_onnx()
