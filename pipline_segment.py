import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import numpy as np
import time
import os
from tqdm import tqdm

# --- 1. 定义 PCA 处理器 ---
class DINOPCASegmenter:
    def __init__(self, model_name="dinov3_vits16", device="cuda"):
        self.device = device
        print(f"[*] Loading {model_name}...")
        self.local_repo_dir = '/home/wayrobo/0_code/dinov3'
        self.weights_backbone = '/home/wayrobo/0_code/dinov3/dinov3/models/dinov3_vits16_pretrain_lvd1689m-08c60483.pth'
        self.model = torch.hub.load(self.local_repo_dir, model_name, source='local', weights=self.weights_backbone)
        self.model.to(device).eval()
        
        # ViT-S/16 默认 patch_size=16
        self.patch_size = 16
        self.transform = T.Compose([
            T.Resize(512, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(512),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    @torch.no_grad()
    def process_image(self, img_path):
        img_raw = Image.open(img_path).convert('RGB')
        img_resized = img_raw.resize((512, 512))
        input_tensor = self.transform(img_raw).unsqueeze(0).to(self.device)

        # 计时开始
        if self.device == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        else:
            start_time = time.time()

        # 1. 提取特征
        # 获取最后一层 patch tokens: [1, 1024, 384] (对于 512/16=32, 32*32=1024)
        features = self.model.get_intermediate_layers(input_tensor, n=1)[0]
        b, n, c = features.shape
        h_feat = w_feat = int(n**0.5)

        # 2. PCA 计算
        # 展平特征 [N, C] -> [1024, 384]
        feat_flat = features.squeeze(0) 
        feat_centered = feat_flat - feat_flat.mean(dim=0)
        
        # 使用低秩分解计算前 3 个主成分
        U, S, V = torch.pca_lowrank(feat_centered, q=3)
        pca_feat = torch.matmul(feat_centered, V[:, :3]) # [1024, 3]

        # 3. 归一化到 0-255 并重构图像
        for i in range(3):
            pca_feat[:, i] = (pca_feat[:, i] - pca_feat[:, i].min()) / \
                             (pca_feat[:, i].max() - pca_feat[:, i].min() + 1e-8)
        
        pca_img = pca_feat.reshape(h_feat, w_feat, 3).cpu().numpy()
        pca_img = (pca_img * 255).astype(np.uint8)
        
        # 计时结束
        if self.device == "cuda":
            end_event.record()
            torch.cuda.synchronize()
            latency = start_event.elapsed_time(end_event)
        else:
            latency = (time.time() - start_time) * 1000

        # 上采样回 512x512
        pca_res = Image.fromarray(pca_img).resize((512, 512), resample=Image.NEAREST)
        
        # 拼接原图和结果进行保存
        combined = Image.new('RGB', (1024, 512))
        combined.paste(img_resized, (0, 0))
        combined.paste(pca_res, (512, 0))
        
        return combined, latency

# --- 2. 批量运行 ---
def main():
    img_dir = "./dataset/dinov3_test"  # 图片输入路径
    output_dir = "./pca_results"
    os.makedirs(output_dir, exist_ok=True)
    
    segmenter = DINOPCASegmenter(model_name="dinov3_vits16")
    
    img_list = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    latencies = []

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    print(f"[*] Processing {len(img_list)} images...")
    for img_name in tqdm(img_list):
        res_img, latency = segmenter.process_image(os.path.join(img_dir, img_name))
        res_img.save(os.path.join(output_dir, f"pca_{img_name}"))
        latencies.append(latency)

    # 统计
    peak_mem = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
    print("\n" + "="*40)
    print(f"PCA Segmentation Report")
    print("-" * 40)
    print(f"Avg Latency (Backbone + PCA): {np.mean(latencies):.2f} ms")
    print(f"Peak GPU Memory:             {peak_mem:.2f} MB")
    print(f"Results saved to:            {output_dir}")
    print("="*40)

if __name__ == "__main__":
    main()