import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
import os
import cv2
from tqdm import tqdm

# --- 🔧 配置区域 ---
# 请设置 DATA_ROOT 为包含 'Rellis_3D_pylon_camera_node' 的上一级目录
# 例如，如果文件夹在 /home/user/data/Rellis_3D_pylon_camera_node
# 那么 DATA_ROOT 应该是 /home/user/data
DATA_ROOT = '/home/wayrobo/.cache/datasets/RELLIS3D' 

# 预训练权重配置
WEIGHTS_PATH = './pretrained/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth'
REPO_DIR = '/home/wayrobo/0_code/dinov3'
MODEL_NAME = 'dinov3_convnext_tiny'
NUM_CLASSES = 7 
BATCH_SIZE = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class DiceLoss(nn.Module):
    def __init__(self, num_classes, ignore_index=255):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        # inputs: [B, C, H, W] (Logits)
        # targets: [B, H, W] (Indices)
        
        inputs = F.softmax(inputs, dim=1)
        
        # One-hot 编码 target
        targets_one_hot = F.one_hot(targets.clamp(0, self.num_classes-1), num_classes=self.num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float() # [B, C, H, W]
        
        # 创建 mask 忽略 ignore_index
        valid_mask = (targets != self.ignore_index).unsqueeze(1).float()
        
        intersection = (inputs * targets_one_hot * valid_mask).sum(dim=(2, 3))
        union = (inputs * valid_mask).sum(dim=(2, 3)) + (targets_one_hot * valid_mask).sum(dim=(2, 3))
        
        dice = 2. * intersection / (union + 1e-8)
        return 1 - dice.mean()

# --- 1. 数据集定义 (针对分离式目录结构) ---
class RellisGolfSplitDataset(Dataset):
    def __init__(self, data_root, transform=None):
        self.data_root = data_root
        self.transform = transform
        self.image_paths = []
        self.mask_paths = []
        
        # 1. 定义两个独立的根入口
        # 根据您的结构：
        # 图片位于: root / Rellis_3D_pylon_camera_node / Rellis-3D / {seq} / pylon_camera_node
        # 标签位于: root / Rellis_3D_pylon_camera_node_label_id / Rellis-3D / {seq} / pylon_camera_node_label_id
        
        self.img_base_dir = os.path.join(data_root, 'Rellis_3D_pylon_camera_node', 'Rellis-3D')
        self.mask_base_dir = os.path.join(data_root, 'Rellis_3D_pylon_camera_node_label_id', 'Rellis-3D')

        print(f"[*] 图片基准路径: {self.img_base_dir}")
        print(f"[*] 标签基准路径: {self.mask_base_dir}")

        if not os.path.exists(self.img_base_dir):
            raise FileNotFoundError(f"找不到图片目录: {self.img_base_dir}")
        if not os.path.exists(self.mask_base_dir):
            raise FileNotFoundError(f"找不到标签目录: {self.mask_base_dir}")

        # 2. 遍历序列文件夹 (00000, 00001...)
        # 我们以图片目录为准来寻找序列
        seq_list = sorted([d for d in os.listdir(self.img_base_dir) if os.path.isdir(os.path.join(self.img_base_dir, d))])
        
        if not seq_list:
            raise RuntimeError(f"在 {self.img_base_dir} 下未找到任何序列文件夹(00000, 00001...)")

        for seq_name in seq_list:
            # 构造该序列的具体图片和标签目录
            # 图片: .../00000/pylon_camera_node
            # 标签: .../00000/pylon_camera_node_label_id
            
            seq_img_dir = os.path.join(self.img_base_dir, seq_name, 'pylon_camera_node')
            seq_mask_dir = os.path.join(self.mask_base_dir, seq_name, 'pylon_camera_node_label_id')
            
            # 检查两个目录是否都存在
            if not os.path.exists(seq_img_dir):
                continue
            if not os.path.exists(seq_mask_dir):
                print(f"[!] 警告: 序列 {seq_name} 有图片但无标签目录，跳过。")
                continue

            # 3. 匹配文件
            fnames = sorted([f for f in os.listdir(seq_img_dir) if f.endswith('.jpg')])
            
            count = 0
            for fname in fnames:
                img_path = os.path.join(seq_img_dir, fname)
                
                # 构造对应的 mask 文件名 (.jpg -> .png)
                mask_name = fname.replace('.jpg', '.png')
                mask_path = os.path.join(seq_mask_dir, mask_name)
                
                if os.path.exists(mask_path):
                    self.image_paths.append(img_path)
                    self.mask_paths.append(mask_path)
                    count += 1
            
            # print(f"    序列 {seq_name}: 加载 {count} 张")

        if len(self.image_paths) == 0:
            raise RuntimeError("未找到任何匹配的图片和标签！请检查文件名后缀是否匹配 (.jpg vs .png)")
            
        print(f"[*] 扫描完毕: 总计 {len(self.image_paths)} 对数据")
        print(f"[*] 示例图片: {self.image_paths[0]}")
        print(f"[*] 示例标签: {self.mask_paths[0]}")

        # 1.2 建立 ID 映射表 (保持不变)
        self.mapping = np.ones(256, dtype=np.uint8) * 255
        self.mapping[2] = 0   # Grass -> Field
        self.mapping[1] = 0   # Dirt -> Field
        self.mapping[23] = 1  # Concrete -> Road
        self.mapping[21] = 1  # Asphalt -> Road
        self.mapping[33] = 2   # Mud -> Sand
        self.mapping[31] = 3  # Puddle -> Water
        self.mapping[6] = 3  # Water
        self.mapping[4] = 4   # Tree -> Tree
        self.mapping[19] = 4  # Bush -> Tree
        self.mapping[17] = 5  # Person
        self.mapping[27] = 6   # Barrier -> Obstacle
        self.mapping[18] = 6  # Fence -> Obstacle
        self.mapping[5] = 6  # Pole -> Obstacle
        self.mapping[7] = 255 # Sky

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        
        image = cv2.imread(img_path)
        if image is None: return self.__getitem__((idx + 1) % len(self))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None: return self.__getitem__((idx + 1) % len(self))

        mask_mapped = self.mapping[mask]
        
        if self.transform:
            image = self.transform(image)
            mask_tensor = torch.from_numpy(mask_mapped).long().unsqueeze(0)
            mask_resized = transforms.functional.resize(
                mask_tensor, (512, 512), 
                interpolation=transforms.InterpolationMode.NEAREST
            ).squeeze(0)
        
        return image, mask_resized

# --- 2. 模型定义 (保持不变) ---
class DINOv3Segmenter(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        print(f"[*] Loading Backbone from {REPO_DIR}...")
        self.backbone = torch.hub.load(REPO_DIR, MODEL_NAME, source='local', pretrained=False)
        
        if os.path.exists(WEIGHTS_PATH):
            checkpoint = torch.load(WEIGHTS_PATH, map_location='cpu')
            if 'model' in checkpoint: state_dict = checkpoint['model']
            elif 'teacher' in checkpoint: state_dict = checkpoint['teacher']
            else: state_dict = checkpoint
            state_dict = {k.replace("module.", "").replace("backbone.", ""): v for k, v in state_dict.items()}
            self.backbone.load_state_dict(state_dict, strict=False)
            print("[*] Weights loaded.")
        
        #for param in self.backbone.parameters():
        #    param.requires_grad = False
            
        #self.head = nn.Conv2d(768, num_classes, kernel_size=1)
        # 🟢 新的 Head (增加空间感知能力)
        self.head = nn.Sequential(
            # 1. 3x3 卷积，融合周围特征
            nn.Conv2d(768, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # 2. 1x1 卷积，输出分类
            nn.Dropout(0.1), # 防止过拟合
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

    def forward(self, x):
        features = self.backbone.forward_features(x)
        if isinstance(features, dict):
            features = features.get('x_norm_patchtokens', list(features.values())[-1])
        if features.dim() == 3:
            B, N, C = features.shape
            size = int(N ** 0.5)
            features = features.view(B, size, size, C).permute(0, 3, 1, 2)
        logits = self.head(features)
        return F.interpolate(logits, size=x.shape[2:], mode='bilinear', align_corners=False)

# --- 3. 训练主流程 ---
def train_rellis_split():
    print("🚀 开始 RELLIS 训练流程 (分离目录版)...")
    
    img_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    try:
        # 使用新的 SplitDataset 类
        full_dataset = RellisGolfSplitDataset(DATA_ROOT, transform=img_transform)
    except Exception as e:
        print(f"❌ 数据集初始化失败: {e}")
        # 打印一下期望的路径帮助调试
        print(f"请检查 DATA_ROOT: {DATA_ROOT} 下是否存在 Rellis_3D_pylon_camera_node 文件夹")
        return
    
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    print(f"[*] 训练集: {len(train_set)} | 验证集: {len(val_set)}")

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = DINOv3Segmenter(NUM_CLASSES).to(DEVICE)
    optimizer = optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': 0.00001},
        {'params': model.head.parameters(), 'lr': 0.001}], weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss(ignore_index=255) 
    # ⚠️ 【关键修改】添加 Class Weights 以解决 "全图是树" 的问题
    # 0:Grass, 1:Road, 2:Sand, 3:Water, 4:Tree, 5:Person, 6:Obstacle
    # 降低 Tree(4) 的权重，大幅提升稀有类别的权重
    class_weights = torch.tensor([3.0, 1.5, 3.0, 3.0, 0.8, 10.0, 5.0]).to(DEVICE)
    
    criterion_ce = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)
    criterion_dice = DiceLoss(NUM_CLASSES, ignore_index=255).to(DEVICE)

    NUM_EPOCHS = 15
    best_loss = float('inf')

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        
        for images, masks in pbar:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss_ce = criterion_ce(outputs, masks)
            loss_dice = criterion_dice(outputs, masks)
            loss = 0.5 * loss_ce + 0.5 * loss_dice
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = train_loss / len(train_loader)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), './pretrained/rellis_golf_best.pth')
            print("💾 Best Model Saved!")

if __name__ == "__main__":
    train_rellis_split()
