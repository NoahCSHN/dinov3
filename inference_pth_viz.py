import os
import time
import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
import math

class DINOv3InferenceWrapper(nn.Module):
    def __init__(self, local_repo_dir, model_name, weights_path, device='cuda'):
        super().__init__()
        self.device = device
        print(f"[*] Loading PyTorch Model: {model_name}")
        
        # 1. 加载模型结构
        self.model = torch.hub.load(
            local_repo_dir, 
            model_name, 
            source='local', 
            pretrained=False
        )
        
        # 2. 加载权重 (Robust Loading)
        if os.path.exists(weights_path):
            print(f"[*] Loading weights from: {weights_path}")
            # 兼容 torch.load 的新旧版本安全策略
            try:
                checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
            except TypeError:
                checkpoint = torch.load(weights_path, map_location='cpu')

            # 提取 state_dict
            if 'model' in checkpoint: state_dict = checkpoint['model']
            elif 'teacher' in checkpoint: state_dict = checkpoint['teacher']
            else: state_dict = checkpoint
            
            # 清洗前缀 (module., backbone. 等)
            state_dict = {k.replace("module.", "").replace("backbone.", ""): v for k, v in state_dict.items()}
            
            # 加载
            msg = self.model.load_state_dict(state_dict, strict=False)
            print(f"[*] Weights loaded. Missing keys: {len(msg.missing_keys)}")
        else:
            raise FileNotFoundError(f"权重文件不存在: {weights_path}")
            
        self.model.to(device)
        self.model.eval()

    def forward(self, x):
        # x: [B, 3, H, W]
        # 使用 forward_features 提取特征
        out = self.model.forward_features(x)
        
        # --- DINOv3 特有的输出处理 ---
        # 1. 如果是字典，提取 patch tokens
        if isinstance(out, dict):
            # 优先找 patch tokens
            if 'x_norm_patchtokens' in out:
                tokens = out['x_norm_patchtokens']
            else:
                # 找不到就取最后一个 value
                tokens = list(out.values())[-1]
        else:
            tokens = out

        # 2. 如果是序列 [B, N, C]，还原为空间图 [B, C, H, W]
        if tokens.dim() == 3:
            B, N, C = tokens.shape
            # 假设是方形输入，计算 H, W
            # e.g., 512输入 -> 16x16 grid -> N=256
            grid_size = int(math.sqrt(N))
            
            # Reshape: [B, N, C] -> [B, H, W, C]
            tokens = tokens.view(B, grid_size, grid_size, C)
            # Permute: [B, H, W, C] -> [B, C, H, W]
            feature_map = tokens.permute(0, 3, 1, 2)
            return feature_map
        
        # 如果已经是 4D，直接返回
        return tokens

class Visualizer:
    def __init__(self, input_size=(512, 512)):
        self.h, self.w = input_size
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def preprocess(self, img_path):
        """OpenCV 读取 -> Tensor"""
        img_raw = cv2.imread(img_path)
        if img_raw is None: raise ValueError(f"无法读取: {img_path}")
        
        # Resize
        img = cv2.resize(img_raw, (self.w, self.h))
        
        # BGR -> RGB & Normalize
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_norm = (img_rgb - self.mean) / self.std
        
        # HWC -> NCHW
        img_input = img_norm.transpose(2, 0, 1)[None, ...] # [1, 3, H, W]
        
        # 转 PyTorch Tensor
        tensor = torch.from_numpy(np.ascontiguousarray(img_input)).float()
        return tensor, cv2.resize(img_raw, (self.w, self.h))

    def compute_pca(self, feature_tensor):
        """
        Input: PyTorch Tensor [1, C, H, W]
        Output: Numpy Image [H, W, 3]
        """
        # 转 numpy
        if isinstance(feature_tensor, torch.Tensor):
            feature_map = feature_tensor.detach().cpu().numpy()
        else:
            feature_map = feature_tensor

        B, C, H, W = feature_map.shape
        
        # [B, C, H, W] -> [H*W, C]
        features = feature_map.transpose(0, 2, 3, 1).reshape(-1, C)
        
        # PCA
        pca = PCA(n_components=3)
        pca_features = pca.fit_transform(features)
        
        # Min-Max Norm
        f_min = pca_features.min(axis=0)
        f_max = pca_features.max(axis=0)
        pca_features = (pca_features - f_min) / (f_max - f_min + 1e-5)
        pca_features = (pca_features * 255).astype(np.uint8)
        
        # Reshape back
        pca_img = pca_features.reshape(H, W, 3)
        
        # Upsample (使用 Cubic 插值获得平滑效果)
        pca_img_resized = cv2.resize(pca_img, (self.w, self.h), interpolation=cv2.INTER_CUBIC)
        
        return pca_img_resized

def main():
    parser = argparse.ArgumentParser()
    # 路径配置
    parser.add_argument("--repo_dir", type=str, default='/home/wayrobo/0_code/dinov3', help="DINOv3 代码根目录")
    parser.add_argument("--weights", type=str, required=True, help=".pth 权重文件路径")
    parser.add_argument("--input_dir", type=str, required=True, help="图片文件夹")
    parser.add_argument("--output_dir", type=str, default="results_pytorch", help="输出文件夹")
    parser.add_argument("--model_name", type=str, default="dinov3_convnext_tiny", help="模型名称")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Running on device: {device}")

    # 1. 初始化模型
    try:
        model = DINOv3InferenceWrapper(args.repo_dir, args.model_name, args.weights, device)
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    viz = Visualizer()
    img_files = [f for f in os.listdir(args.input_dir) if f.lower().endswith(('.jpg', '.png'))]
    print(f"[*] 找到 {len(img_files)} 张图片")
    
    # Warmup
    dummy = torch.randn(1, 3, 512, 512).to(device)
    with torch.no_grad():
        model(dummy)

    total_time = 0
    
    # 2. 推理循环
    for idx, img_name in enumerate(img_files):
        img_path = os.path.join(args.input_dir, img_name)
        
        # 预处理
        tensor_in, img_vis = viz.preprocess(img_path)
        tensor_in = tensor_in.to(device)
        
        # 推理 & 计时
        torch.cuda.synchronize()
        t0 = time.time()
        
        with torch.no_grad():
            features = model(tensor_in)
            
        torch.cuda.synchronize()
        t1 = time.time()
        latency = (t1 - t0) * 1000
        total_time += latency
        
        # 可视化
        pca_vis = viz.compute_pca(features)
        
        # 拼接 (BGR for OpenCV)
        pca_vis_bgr = cv2.cvtColor(pca_vis, cv2.COLOR_RGB2BGR)
        combined = cv2.hconcat([img_vis, pca_vis_bgr])
        
        text = f"PyTorch (Raw): {latency:.2f}ms | Shape: {list(features.shape)}"
        cv2.putText(combined, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imwrite(os.path.join(args.output_dir, img_name), combined)
        print(f"[{idx+1}/{len(img_files)}] {img_name:<20} | {latency:.2f} ms")

    print("="*50)
    print(f"📊 PyTorch 平均时延: {total_time/len(img_files):.2f} ms")
    print("="*50)

if __name__ == "__main__":
    main()
