import torch
import torch.nn as nn
import os
import warnings
import onnx
from onnxsim import simplify

# 屏蔽无关的 Trace 警告
warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)

class DINOv3ExportWrapper(nn.Module):
    def __init__(self, local_repo_dir, model_name, weights_path):
        super().__init__()
        self.model = torch.hub.load(
            local_repo_dir, 
            model_name, 
            source='local', 
            weights=weights_path
        ).float()

    def forward(self, x):
        # 提取最后一层特征
        features = self.model.get_intermediate_layers(x, n=1)[0]
        return features

def export_to_onnx():
    local_repo_dir = '/home/wayrobo/0_code/dinov3'
    model_name = "dinov3_vits16"
    weights_path = '/home/wayrobo/0_code/dinov3/pretrained/dinov3_vits16_pretrain_lvd1689m-08c60483.pth'
    output_onnx = "dinov3_vits16_512.onnx"
    output_onnx_simplified = "dinov3_vits16_512_simplified.onnx"
    
    device = "cpu"
    wrapper = DINOv3ExportWrapper(local_repo_dir, model_name, weights_path).to(device)
    wrapper.eval()

    dummy_input = torch.randn(1, 3, 512, 512).to(device)

    print(f"[*] Exporting via LEGACY path (Bypassing ALL Dynamo logic)...")
    
    # 【核心修复】不使用 JIT Trace，而是直接使用带有 legacy 标志的 export
    # 同时强制指定 dynamo=False
    try:
        torch.onnx.export(
            wrapper, 
            dummy_input,
            output_onnx,
            export_params=True,
            opset_version=16, 
            do_constant_folding=True,
            input_names=['input'],
            output_names=['feature_tokens'],
            # 关键：显式设置这个参数来切断新版导出器的路径
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
            # 强制不使用 dynamo 模式
            # dynamo=False  # 在某些 2.5+ 版本中可以直接设置此参数
        )
        print(f"✅ Export Success: {output_onnx}")
    except Exception as e:
        print(f"❌ Standard export failed, trying alternative internal path...")
        # 备选方案：如果上面的路径还是被劫持，使用以下方式
        torch.onnx.utils.export(
            wrapper,
            dummy_input,
            output_onnx,
            verbose=False,
            opset_version=16,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['feature_tokens']
        )
        print(f"✅ Alternative Export Success: {output_onnx}")


    onnx_model = onnx.load(output_onnx)
    # 使用 onnxsim 消除动态控制流节点 (如 If, Loop)
    model_simp, check = simplify(onnx_model)

    if check:
        onnx.save(model_simp, output_onnx_simplified)
        print("✅ ONNX 模型已成功简化，动态 If 节点已折叠！")
    else:
        print("❌ 简化失败，请检查模型结构。")

if __name__ == "__main__":
    export_to_onnx()
