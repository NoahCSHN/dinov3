import torch
import os
import cv2
import numpy as np
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as TVT
from tqdm import tqdm

# 官方 80 个提示词模板 (参考自 Notebook)
PROMPT_TEMPLATES = [
    "a photo of a {}.", "a rendering of a {}.", "a cropped photo of the {}.",
    "the photo of a {}.", "a photo of my {}.", "a photo of the cool {}.",
    "a close-up photo of a {}.", "a bright photo of the {}.",
    "a dark photo of the {}.", "a blurry photo of the {}."
    # ... (为了简洁此处省略，建议保留你代码中定义的列表)
]

class GolfCourseAnalyzer:
    def __init__(self, repo_dir, model_name, backbone_w, dinotxt_w, labels):
        self.device = "cuda"
        self.labels = labels
        
        # 1. 加载模型 (官方 Hub 方式)
        self.model, self.tokenizer = torch.hub.load(
            repo_dir, model_name, weights=dinotxt_w, 
            backbone_weights=backbone_w, source='local'
        )
        self.model.to(self.device).half().eval()
        
        # 2. 预处理 (官方推荐: 短边缩放至 512 以提升小物体识别)
        self.resize_size = 512 
        self.transform = TVT.Compose([
            TVT.Resize(self.resize_size, interpolation=TVT.InterpolationMode.BICUBIC),
            TVT.ToTensor(),
            TVT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # 3. 准备文本特征 (包含 Prompt Ensemble 和 1024 切片)
        self.text_features = self._prepare_text_features()

    @torch.no_grad()
    def _prepare_text_features(self):
        all_class_features = []
        print("🔤 正在生成文本集成特征...")
        for label in self.labels:
            texts = [temp.format(label) for temp in PROMPT_TEMPLATES]
            tokens = self.tokenizer.tokenize(texts).to(self.device)
            
            with torch.amp.autocast('cuda'):
                # 得到 [num_templates, 2048]
                feats = self.model.encode_text(tokens)
                # 核心修复：切片取后半部分对齐 Patch
                feats = feats[:, feats.shape[1] // 2 :] 
                # 官方双重归一化逻辑
                feats = F.normalize(feats, p=2, dim=-1)
                feats = feats.mean(dim=0)
                feats = F.normalize(feats, p=2, dim=-1)
                all_class_features.append(feats)
        return torch.stack(all_class_features)

    @torch.no_grad()
    def analyze_image(self, img_path):
        img_pil = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(img_pil).unsqueeze(0).to(self.device).half()
        
        # 记录原始尺寸用于 HeatMap 叠加
        orig_np = np.array(img_pil)
        h_orig, w_orig = orig_np.shape[:2]

        with torch.amp.autocast('cuda'):
            # 采用官方 predict_whole 逻辑提取特征
            _, _, patch_tokens = self.model.visual_model.get_class_and_patch_tokens(img_tensor)
            patch_tokens = F.normalize(patch_tokens, p=2, dim=-1) # [1, 196, 1024]
            
            # 计算相似度 [1, 196, num_labels]
            similarity = patch_tokens @ self.text_features.T
            
            # 提取相似度图 (14x14)
            sim_map = similarity.reshape(1, 14, 14, -1).permute(0, 3, 1, 2) # [1, num_labels, 14, 14]
            
            # 上采样到原图尺寸
            sim_map_resized = F.interpolate(sim_map, size=(h_orig, w_orig), mode='bilinear', align_corners=False)
            sim_map_np = sim_map_resized.squeeze(0).cpu().float().numpy()

        return sim_map_np, orig_np

    def batch_process(self, input_dir, output_dir, limit=100):
        """
        批量处理入口：生成分割图与热力图诊断。
        通过热力图，我们可以直接观察模型对“码牌”类别的原始响应强度。
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 1. 获取图片列表
        files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        files.sort()
        files = files[:limit]
        
        print(f"🚀 开始批量分析 {len(files)} 张图片，诊断目录: {output_dir}")
        
        for i, filename in enumerate(tqdm(files, desc="Processing")):
            try:
                img_path = os.path.join(input_dir, filename)
                
                # 2. 执行分析：获取所有类别的相似度图 [num_classes, H, W]
                # 这是基于官方 predict_whole 逻辑的实现
                sim_maps, original_img = self.analyze_image(img_path)
                
                # 3. 调用诊断保存函数
                # 它会保存：1. 最终分割图  2. 码牌专项热力图
                self.save_diagnostic_plots(sim_maps, original_img, output_dir, filename)
                
                # 4. 显存管理（针对 WSL2 8GB 环境）
                if (i + 1) % 10 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
                    
            except Exception as e:
                print(f"❌ 无法处理 {filename}: {str(e)}")
                continue

        print(f"✅ 任务完成。请检查 {output_dir} 中的 heatmap_*.png 文件。")

    def save_diagnostic_plots(self, sim_maps, original_img, save_dir, filename):
        """生成并保存热力图，验证模型对特定目标的感知力"""
        h, w = original_img.shape[:2]
        
        # --- 诊断 A: 码牌热力图 ---
        # 假设 "orange yardage marker" 是 queries 中的第 4 个索引
        marker_idx = 4 
        heatmap_raw = sim_maps[marker_idx]
        
        # 归一化到 0-255 以便 OpenCV 伪彩色映射
        heatmap_norm = cv2.normalize(heatmap_raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
        
        # 将热力图叠加在原图上 (50% 透明度)
        overlay = cv2.addWeighted(cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR), 0.6, heatmap_color, 0.4, 0)
        
        # --- 诊断 B: 最终分割结果 (Argmax) ---
        mask = np.argmax(sim_maps, axis=0)
        segmentation = original_img.copy()
        colors = [[0, 255, 0], [0, 255, 255], [255, 0, 0], [255, 255, 255], [255, 0, 255], [128, 128, 128]]
        
        for idx, color in enumerate(colors):
            if idx < len(self.labels):
                segmentation[mask == idx] = segmentation[mask == idx] * 0.5 + np.array(color) * 0.5
        
        # 保存对比图：左边是原图叠加分割，右边是码牌热力图
        combined = np.hstack([cv2.cvtColor(segmentation, cv2.COLOR_RGB2BGR), overlay])
        cv2.imwrite(os.path.join(save_dir, f"diag_{filename}"), combined)

# --- 修改后的主配置 ---
if __name__ == "__main__":
    CONFIG = {
        "repo_dir": "/home/wayrobo/0_code/dinov3",
        "model_name": "dinov3_vitl16_dinotxt_tet1280d20h24l",
        "backbone_w": "/home/wayrobo/0_code/dinov3/pretrained/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
        "dinotxt_w": "/home/wayrobo/0_code/dinov3/pretrained/dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth",
        "input_pics": "/home/wayrobo/0_code/dinov3/dataset/dinov3_test",
        "output_dir": "./golf_diagnostic",
        "queries": [
            "green grass fairway",       # 0
            "sand bunker",               # 1
            "water pond",                # 2
            "golf net",                  # 3
            "orange yardage marker sign",# 4: 强化描述
            "trees background"           # 5
        ]
    }

    analyzer = GolfCourseAnalyzer(
        CONFIG["repo_dir"], CONFIG["model_name"], 
        CONFIG["backbone_w"], 
        CONFIG["dinotxt_w"], 
        CONFIG["queries"]
    )
    
    # 批量处理逻辑中调用新的诊断函数
    # ... (保持 batch_process 结构，调用 analyzer.analyze_image 和 analyzer.save_diagnostic_plots)
    analyzer.batch_process(CONFIG["input_pics"], CONFIG["output_dir"])

