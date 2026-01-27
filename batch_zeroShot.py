import torch
import os
import cv2
import numpy as np
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as TVT
from tqdm import tqdm

# ================= 决策配置区 =================
CONFIG = {
    "repo_dir": "/home/wayrobo/0_code/dinov3",
    "model_name": "dinov3_vitl16_dinotxt_tet1280d20h24l",
    "backbone_w": "/home/wayrobo/0_code/dinov3/pretrained/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
    "dinotxt_w": "/home/wayrobo/0_code/dinov3/pretrained/dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth",
    "input_pics": "/home/wayrobo/0_code/dinov3/dataset/dinov3_test",
    "output_dir": "./golf_segmentation_results",
    # 【优化项 1】强化语义描述，利用 FP32 的高精度对齐
    "queries": [
        "plain green grass ground",                   # 0: 草地 (调低语义强度)
        "sand bunker",                                # 1
        "water pond",                                 # 2
        "protective net",                             # 3
        "rectangular sign with number",               # 4: 码牌 (强化特征描述)
        "dark trees"                                  # 5
    ],
    # 【优化项 2】决策增益 (Bias)
    # 根据热力图表现，我们需要给码牌约 0.15 - 0.20 的“战力补偿”
    "marker_bias": 0.00, 
    "grass_penalty": 0.00
}

class GolfResultAnalyzer:
    def __init__(self):
        self.device = "cuda"
        print(f"🏗️  正在加载全精度 FP32 模型以保证分类准确性...")
        self.model, self.tokenizer = torch.hub.load(
            CONFIG["repo_dir"], CONFIG["model_name"], 
            weights=CONFIG["dinotxt_w"], backbone_weights=CONFIG["backbone_w"], source='local'
        )
        self.model.to(self.device).float().eval() # 坚持使用 Float32
        
        self.transform = TVT.Compose([
            TVT.Resize(512, interpolation=TVT.InterpolationMode.BICUBIC),
            TVT.ToTensor(),
            TVT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.text_features = self._prepare_text_features()

    @torch.no_grad()
    def _prepare_text_features(self):
        all_feats = []
        templates = ["a photo of a {}.", "a close-up of {}.", "{} in a golf course."]
        for label in CONFIG["queries"]:
            texts = [t.format(label) for t in templates]
            tokens = self.tokenizer.tokenize(texts).to(self.device)
            feats = self.model.encode_text(tokens).float()
            # 官方 1024 维切片逻辑
            feats = feats[:, feats.shape[1] // 2 :]
            feats = F.normalize(feats, p=2, dim=-1, eps=1e-5)
            all_feats.append(F.normalize(feats.mean(0), p=2, dim=-1, eps=1e-5))
        return torch.stack(all_feats)

    @torch.no_grad()
    def analyze_image(self, img_path):
        img_pil = Image.open(img_path).convert("RGB")
        w_orig, h_orig = img_pil.size
        img_tensor = self.transform(img_pil).unsqueeze(0).to(self.device).float()
        
        # 提取视觉特征
        _, _, patch_tokens = self.model.visual_model.get_class_and_patch_tokens(img_tensor)
        patch_tokens = F.normalize(patch_tokens.squeeze(0).float(), p=2, dim=-1, eps=1e-5)
        
        # 计算原始相似度
        similarity = patch_tokens @ self.text_features.T # [N, 6]
        
        # --- 实时分数监控 (解决全绿问题的核心) ---
        idx_marker = 4
        marker_max = similarity[:, idx_marker].max().item()
        # 找到码牌响应最高点对应的草地分值
        grass_at_marker = similarity[similarity[:, idx_marker].argmax(), 0].item()
        
        # 【关键日志】输出这个信息能帮你微调 bias
        print(f"DEBUG [{os.path.basename(img_path)}] -> 码牌最高分: {marker_max:.4f} | 竞争草地分: {grass_at_marker:.4f} | 差距: {grass_at_marker - marker_max:.4f}")

        # --- 竞争补偿逻辑 ---
        bias = torch.zeros(similarity.shape[-1], device=self.device)
        bias[0] = CONFIG["grass_penalty"] # 压低草地
        bias[4] = CONFIG["marker_bias"]   # 提升码牌
        
        # 最终决策 (使用 30.0 缩放因子增强类别边界)
        final_scores = (similarity + bias) * 30.0
        grid_h, grid_w = img_tensor.shape[2]//16, img_tensor.shape[3]//16
        mask = final_scores.argmax(dim=-1).reshape(grid_h, grid_w).cpu().numpy()
        
        return mask, np.array(img_pil)

    def save_final_result(self, mask, original_img, save_dir, filename):
        h, w = original_img.shape[:2]
        mask_res = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        
        # BGR 颜色定义：0:绿(草), 4:紫(码牌)
        colors = [[0,255,0], [0,255,255], [255,0,0], [255,255,255], [255,0,255], [128,128,128]]
        
        res_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)
        overlay = res_img.copy()
        
        for i, color in enumerate(colors):
            if i < len(CONFIG["queries"]):
                region = mask_res == i
                if np.any(region):
                    overlay[region] = cv2.addWeighted(overlay[region], 0.4, np.full_like(overlay[region], color), 0.6, 0).squeeze()
        
        cv2.imwrite(os.path.join(save_dir, f"result_{filename}"), overlay)

    def run(self):
        if not os.path.exists(CONFIG["output_dir"]): os.makedirs(CONFIG["output_dir"])
        files = sorted([f for f in os.listdir(CONFIG["input_pics"]) if f.lower().endswith(('.png', '.jpg'))])[:20]
        for f in tqdm(files):
            mask, orig = self.analyze_image(os.path.join(CONFIG["input_pics"], f))
            self.save_final_result(mask, orig, CONFIG["output_dir"], f)

if __name__ == "__main__":
    analyzer = GolfResultAnalyzer()
    analyzer.run()
