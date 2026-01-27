import torch
import os
import cv2
import numpy as np
import gc
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as TVT
from tqdm import tqdm

# ================= 配置区 =================
CONFIG = {
    "repo_dir": "/home/wayrobo/0_code/dinov3",
    "model_name": "dinov3_vitl16_dinotxt_tet1280d20h24l",
    "backbone_w": "/home/wayrobo/0_code/dinov3/pretrained/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
    "dinotxt_w": "/home/wayrobo/0_code/dinov3/pretrained/dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth",
    "input_pics": "/home/wayrobo/0_code/dinov3/dataset/dinov3_test",
    "output_dir": "./golf_final_results_fp32",
    # 采用简单颜色标签进行感知测试
    "queries": ["grass", "sand", "water", "net", "board with number", "trees"]
}

PROMPT_TEMPLATES = [
    "a photo of a {}.", "a cropped photo of the {}.", "a close-up photo of a {}.",
    "a bright photo of the {}.", "a dark photo of the {}."
]

class GolfCourseAnalyzer:
    def __init__(self):
        self.device = "cuda"
        print(f"🏗️  正在以 FP32 全精度加载 DINOv3 模型...")
        # 1. 加载模型逻辑
        self.model, self.tokenizer = torch.hub.load(
            CONFIG["repo_dir"], CONFIG["model_name"], 
            weights=CONFIG["dinotxt_w"], 
            backbone_weights=CONFIG["backbone_w"], 
            source='local'
        )
        # 2. 【核心修改】强制使用 float() 并移除 half()
        self.model.to(self.device).float().eval() 
        
        self.transform = TVT.Compose([
            TVT.Resize(512, interpolation=TVT.InterpolationMode.BICUBIC),
            TVT.ToTensor(),
            TVT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        self.text_features = self._prepare_text_features()

    @torch.no_grad()
    def _prepare_text_features(self):
        all_feats = []
        print("🔤 正在生成文本特征 (Float32模式)...")
        for label in CONFIG["queries"]:
            texts = [temp.format(label) for temp in PROMPT_TEMPLATES]
            tokens = self.tokenizer.tokenize(texts).to(self.device)
            
            # encode_text 输出现在默认为 Float32
            feats = self.model.encode_text(tokens).float() 
            
            # 切片取后半段 1024 维
            half_dim = feats.shape[1] // 2
            p2_feats = feats[:, half_dim:]
            
            # 增加 eps 防御归一化中的 nan
            p2_feats = F.normalize(p2_feats, p=2, dim=-1, eps=1e-5)
            mean_feat = p2_feats.mean(dim=0)
            final_feat = F.normalize(mean_feat, p=2, dim=-1, eps=1e-5)
            all_feats.append(final_feat)
            
        return torch.stack(all_feats)

    @torch.no_grad()
    def analyze_image(self, img_path):
        img_pil = Image.open(img_path).convert("RGB")
        w_orig, h_orig = img_pil.size
        # 3. 【核心修改】输入张量保持为 float
        img_tensor = self.transform(img_pil).unsqueeze(0).to(self.device).float()
        
        patch_h, patch_w = img_tensor.shape[2] // 16, img_tensor.shape[3] // 16

        # 4. 【核心修改】彻底停用 autocast 加速块
        _, _, patch_tokens = self.model.visual_model.get_class_and_patch_tokens(img_tensor)
        patch_tokens = F.normalize(patch_tokens.float(), p=2, dim=-1, eps=1e-5)
        
        # 计算相似度 [N, num_labels]
        similarity = patch_tokens.squeeze(0) @ self.text_features.T
        
        # 找到码牌响应最强的一个 Patch 索引
        # idx_max = similarity[:, 4].argmax()
        # marker_score = similarity[idx_max, 4].item()
        # grass_score = similarity[idx_max, 0].item()

        # print(f"📊 目标区域对决 -> 码牌相似度: {marker_score:.4f}, 草地相似度: {grass_score:.4f}")
        # print(f"📉 当前差距: {grass_score - marker_score:.4f}")
        # 应用偏置并生成 Mask
        bias = torch.zeros(similarity.shape[-1], device=self.device)
        bias[4] = 0.12 # 略微加大对 Index 4 (red color) 的补偿
        
        final_scores = (similarity + bias) * 20.0
        mask = final_scores.argmax(dim=-1).reshape(patch_h, patch_w).cpu().numpy()
        
        # 准备热力图数据
        sim_map = similarity.reshape(1, patch_h, patch_w, -1).permute(0, 3, 1, 2)
        sim_map_resized = F.interpolate(sim_map, size=(h_orig, w_orig), mode='bilinear').squeeze(0)

        return mask, sim_map_resized.cpu().float().numpy(), np.array(img_pil)

    def save_result(self, mask, sim_map, original_img, save_dir, filename):
        h, w = original_img.shape[:2]
        
        # 处理左侧：分割叠加层
        mask_resized = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        colors_bgr = [[0,255,0], [0,255,255], [255,0,0], [255,255,255], [255,0,255], [128,128,128]]
        
        bgr_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)
        seg_overlay = bgr_img.copy()
        for i, color in enumerate(colors_bgr):
            if i < len(CONFIG["queries"]):
                region = mask_resized == i
                if np.any(region):
                    c = np.array(color, dtype=np.uint8)
                    seg_overlay[region] = cv2.addWeighted(seg_overlay[region], 0.5, np.full_like(seg_overlay[region], c), 0.5, 0).squeeze()
        
        # 5. 【核心修改】输出 100% 纯热力图，排除原图干扰
        heatmap_raw = sim_map[4]
        denom = heatmap_raw.max() - heatmap_raw.min()
        if denom > 1e-6:
            heatmap_norm = cv2.normalize(heatmap_raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        else:
            heatmap_norm = np.zeros_like(heatmap_raw, dtype=np.uint8)
        
        pure_heatmap = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
        
        combined = np.hstack([seg_overlay, pure_heatmap])
        cv2.imwrite(os.path.join(save_dir, f"diag_{filename}"), combined)

    def run_batch(self):
        if not os.path.exists(CONFIG["output_dir"]): os.makedirs(CONFIG["output_dir"])
        files = sorted([f for f in os.listdir(CONFIG["input_pics"]) if f.lower().endswith(('.png', '.jpg'))])[:100]
        for i, filename in enumerate(tqdm(files)):
            try:
                mask, sim_map, orig = self.analyze_image(os.path.join(CONFIG["input_pics"], filename))
                self.save_result(mask, sim_map, orig, CONFIG["output_dir"], filename)
            except Exception as e:
                print(f"❌ Error {filename}: {e}")

if __name__ == "__main__":
    analyzer = GolfCourseAnalyzer()
    analyzer.run_batch()
