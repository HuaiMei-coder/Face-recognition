"""
ResNet人脸识别模型实现
"""

import cv2
import numpy as np
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
import torchvision.models as models
from torch.cuda import amp

class FaceDataset(Dataset):
    """自定义PyTorch数据集类，用于加载彩色人脸图像"""
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 彩色图像处理 (H, W, C)
        image = self.images[idx].astype(np.uint8)  # 确保为uint8格式
        label = self.labels[idx]
        
        if self.transform:
            # 将OpenCV图像(BGR)转换为PIL图像(RGB)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = self.transform(image)
        
        label = torch.tensor(label, dtype=torch.long)
        return image, label

class ResNetFaceRecognizer(nn.Module):
    """
    基于ResNet的人脸识别器，支持彩色图像输入
    """
    def __init__(self, num_classes, epochs=150, batch_size=32, learning_rate=0.001):
        super(ResNetFaceRecognizer, self).__init__()
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes
        
        # 使用预训练的ResNet18作为基础模型
        self.base_model = models.resnet18(pretrained=True)
        
        # 修改第一层以适应彩色输入
        self.base_model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # 替换最后一层全连接层
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        self.to(self.device)
        self.training_time = None
        self.prediction_time = None
        self.train_labels = None
        self.best_val_acc = 0.0
        
    def forward(self, x):
        return self.base_model(x)
    
    def train_model(self, train_images, train_labels, val_images, val_labels):
        """
        训练模型，包含早停和学习率调度
        """
        print("[INFO] 🚀 开始训练ResNet模型...")
        start_time = time.time()
        
        # 步骤1: 数据预处理
        print("\n📊 步骤 1/5: 数据预处理")
        n_samples = train_images.shape[0]
        print(f"[INFO] 训练样本数: {n_samples}")
        print(f"[INFO] 图像尺寸: {train_images.shape[1:]}")
        
        # 创建数据增强
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 创建数据集
        train_dataset = FaceDataset(train_images, train_labels, transform=train_transform)
        val_dataset = FaceDataset(val_images, val_labels, transform=val_transform)
        
        # 创建加权采样器处理类别不平衡
        class_counts = np.bincount(train_labels)
        class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
        sample_weights = class_weights[train_labels]
        sampler = WeightedRandomSampler(
            weights=sample_weights, 
            num_samples=len(sample_weights), 
            replacement=True
        )
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            sampler=sampler,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            pin_memory=True
        )
        
        # 步骤2: 设置优化器和损失函数
        print("\n🎯 步骤 2/5: 设置模型参数")
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # 学习率调度器
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='max', 
            factor=0.5, 
            patience=6,
            verbose=True
        )
        
        # 混合精度训练的梯度缩放器
        scaler = amp.GradScaler(enabled=self.device.type == 'cuda')
        
        # 步骤3: 训练模型
        print("\n🎨 步骤 3/5: 训练模型")
        best_model_wts = None
        no_improve = 0
        patience = 5
        
        for epoch in range(self.epochs):
            self.train()
            running_loss = 0.0
            running_corrects = 0
            
            with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{self.epochs}", unit="batch") as pbar:
                for images, labels in train_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    
                    optimizer.zero_grad()
                    
                    # 混合精度训练
                    with amp.autocast(enabled=self.device.type == 'cuda'):
                        outputs = self(images)
                        loss = criterion(outputs, labels)
                        _, preds = torch.max(outputs, 1)
                    
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    
                    running_loss += loss.item() * images.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                    pbar.update(1)
            
            epoch_loss = running_loss / len(train_dataset)
            epoch_acc = running_corrects.double() / len(train_dataset)
            
            # 验证阶段
            val_acc = self.evaluate(val_loader)
            print(f"训练损失: {epoch_loss:.4f}, 训练准确率: {epoch_acc:.4f}, 验证准确率: {val_acc:.4f}")
            
            # 学习率调度
            scheduler.step(val_acc)
            
            # 早停机制
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                best_model_wts = self.state_dict().copy()
                no_improve = 0
                print(f"🔥 最佳模型更新! 验证准确率: {val_acc:.4f}")
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"⏹️ 早停触发: 验证准确率连续{patience}轮未提升")
                    break
        
        # 加载最佳模型权重
        if best_model_wts:
            self.load_state_dict(best_model_wts)
        
        self.train_labels = train_labels
        self.training_time = time.time() - start_time
        
        print(f"\n✅ 训练完成!")
        print(f"⏱️ 总训练时间: {self.training_time:.2f}秒")
        print(f"💾 训练样本数: {n_samples}")
        print(f"🏆 最佳验证准确率: {self.best_val_acc:.4f}")
    
    def evaluate(self, data_loader):
        """评估模型在验证集上的准确率"""
        self.eval()
        running_corrects = 0
        
        for images, labels in data_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            
            with torch.no_grad():
                outputs = self(images)
                _, preds = torch.max(outputs, 1)
            
            running_corrects += torch.sum(preds == labels.data)
        
        acc = running_corrects.double() / len(data_loader.dataset)
        return acc.item()
    
    def predict(self, test_images):
        """
        预测测试图像的标签
        """
        print("\n🔮 开始预测...")
        start_time = time.time()
        
        n_samples = test_images.shape[0]
        print(f"[INFO] 预测样本数: {n_samples}")
        
        # 预处理
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        test_dataset = FaceDataset(test_images, np.zeros(n_samples, dtype=np.int64), transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        # 预测
        self.eval()
        predicted_labels = []
        
        with torch.no_grad():
            with tqdm(total=len(test_loader), desc="分类预测", unit="batch") as pbar:
                for images, _ in test_loader:
                    images = images.to(self.device)
                    outputs = self(images)
                    _, predicted = torch.max(outputs, 1)
                    predicted_labels.extend(predicted.cpu().numpy())
                    pbar.update(1)
        
        self.prediction_time = time.time() - start_time
        
        print(f"\n✅ 预测完成!")
        print(f"⏱️  预测时间: {self.prediction_time:.2f}秒")
        print(f"⚡  平均预测速度: {n_samples/self.prediction_time:.2f} 张/秒")
        
        return np.array(predicted_labels)
    
    def get_model_info(self):
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        return {
            'num_classes': self.num_classes,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'total_parameters': total_params,
            'training_samples': len(self.train_labels) if self.train_labels is not None else 0,
            'training_time': self.training_time,
            'prediction_time': self.prediction_time,
            'best_val_acc': self.best_val_acc
        }