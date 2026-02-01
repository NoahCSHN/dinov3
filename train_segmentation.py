import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np

# --- 配置 ---
# VOC 有 21 个类别 (0=背景, 1-20=物体)
NUM_CLASSES = 21 
BATCH_SIZE = 8 # 显存不够改小 (Orin Nano 建议 4 或 8)
REPO_DIR = '/home/wayrobo/0_code/dinov3'
WEIGHTS_PATH = './pretrained/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth'
MODEL_NAME = 'dinov3_convnext_tiny'
DEVICE = "cuda"

# --- 定义模型 (保持不变) ---
class DINOv3Segmenter(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        print(f"[*] Loading Backbone...")
        self.backbone = torch.hub.load(REPO_DIR, MODEL_NAME, source='local', pretrained=False)
        
        checkpoint = torch.load(WEIGHTS_PATH, map_location='cpu')
        if 'model' in checkpoint: state_dict = checkpoint['model']
        elif 'teacher' in checkpoint: state_dict = checkpoint['teacher']
        else: state_dict = checkpoint
        state_dict = {k.replace("module.", "").replace("backbone.", ""): v for k, v in state_dict.items()}
        self.backbone.load_state_dict(state_dict, strict=False)
        
        # 冻结 Backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        self.head = nn.Conv2d(768, num_classes, kernel_size=1)

    def forward(self, x):
        features = self.backbone.forward_features(x)
        if isinstance(features, dict):
            features = features.get('x_norm_patchtokens', list(features.values())[-1])
        if features.dim() == 3:
            B, N, C = features.shape
            size = int(N ** 0.5)
            features = features.view(B, size, size, C).permute(0, 3, 1, 2)
            
        logits = self.head(features)
        # 上采样回原图尺寸 (VOC 图片尺寸不固定，我们动态上采样)
        output = F.interpolate(logits, size=x.shape[2:], mode='bilinear', align_corners=False)
        return output

# --- 数据转换 ---
# VOC 的 Mask 是 PIL Image (P模式)，我们需要把它转成 Tensor (Long类型)
# 并且把 255 (边界/忽略区域) 转成 0 或忽略
class MaskToTensor:
    def __call__(self, target):
        target = np.array(target)
        target[target == 255] = 0 # 简单粗暴：把边界当作背景(0)，或者你可以在 Loss 里设置 ignore_index=255
        return torch.from_numpy(target).long()

def train_voc():
    # 1. 自动下载 VOC 2012
    print("正在准备 VOC 2012 数据集 (首次运行会自动下载)...")
    
    # 图像预处理
    img_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Mask 预处理 (Resize 必须用 Nearest 防止产生不存在的类别)
    mask_transform = transforms.Compose([
        transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.NEAREST),
        MaskToTensor()
    ])

    try:
        train_dataset = datasets.VOCSegmentation(
            root='./voc_data', 
            year='2012', 
            image_set='train', 
            download=False, 
            transform=img_transform, 
            target_transform=mask_transform
        )
    except Exception as e:
        print(f"自动下载失败，请尝试手动下载: {e}")
        return

    dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)
    
    print(f"[*] 训练集加载完毕: {len(train_dataset)} 张图片")

    # 2. 初始化模型
    model = DINOv3Segmenter(NUM_CLASSES).to(DEVICE)
    optimizer = optim.AdamW(model.head.parameters(), lr=0.001)
    
    # 忽略 index 255 (VOC 中 255 代表物体边缘的模糊地带，不参与计算 Loss)
    # 如果你在 MaskToTensor 里没处理 255，这里必须加 ignore_index=255
    criterion = nn.CrossEntropyLoss(ignore_index=255) 

    # 3. 训练循环
    for epoch in range(5): # 跑 5 个 epoch 验证一下
        model.train()
        model.backbone.eval()
        
        running_loss = 0.0
        for i, (images, masks) in enumerate(dataloader):
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if i % 10 == 0:
                print(f"Epoch {epoch+1} [{i}/{len(dataloader)}] Loss: {loss.item():.4f}")
        
        print(f"=== Epoch {epoch+1} Avg Loss: {running_loss/len(dataloader):.4f} ===")
        
        # 保存中间结果
        torch.save(model.state_dict(), f'pretrained/voc_segmenter_epoch_{epoch+1}.pth')

if __name__ == "__main__":
    train_voc()
