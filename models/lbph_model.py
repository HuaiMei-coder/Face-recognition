"""
LBPH人脸识别模型实现
"""

import cv2
import numpy as np
import time
from tqdm import tqdm

class EnhancedLBPHRecognizer:
    """
    增强版LBPH识别器
    """
    def __init__(self, radius=1, neighbors=8, grid_x=8, grid_y=8, threshold=100.0):
        """
        初始化LBPH识别器
        """
        self.radius = radius
        self.neighbors = neighbors
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.threshold = threshold
        self.recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=radius, neighbors=neighbors, grid_x=grid_x, grid_y=grid_y, threshold=threshold
        )
        self.train_labels = None
        self.training_time = None
        self.prediction_time = None
        
    def train(self, train_images, train_labels):
        """
        训练LBPH模型
        """
        print("[INFO] 🚀 开始训练LBPH模型...")
        start_time = time.time()
        
        # 步骤1: 数据预处理
        print("\n📊 步骤 1/3: 数据预处理")
        n_samples = train_images.shape[0]
        
        print(f"[INFO] 训练样本数: {n_samples}")
        print(f"[INFO] 图像尺寸: {train_images.shape[1:]}")
        
        # 确保图像为uint8类型
        print("[INFO] 转换图像格式...")
        images_uint8 = train_images.astype(np.uint8)
        
        # 步骤2: 训练LBPH模型
        print("\n🎯 步骤 2/3: 训练LBPH模型")
        print(f"[INFO] 使用参数: radius={self.radius}, neighbors={self.neighbors}, grid_x={self.grid_x}, grid_y={self.grid_y}")
        
        with tqdm(total=100, desc="LBPH训练", unit="%") as pbar:
            self.recognizer.train(list(images_uint8), train_labels.astype(np.int32))
            pbar.update(100)
        
        self.train_labels = train_labels
        self.training_time = time.time() - start_time
        
        print(f"\n✅ 训练完成!")
        print(f"⏱️  总训练时间: {self.training_time:.2f}秒")
        print(f"💾 训练数据形状: {train_images.shape}")
        
    def predict(self, test_images):
        """
        预测测试图像的标签
        """
        if self.recognizer is None:
            raise ValueError("模型必须先训练才能进行预测")
        
        print("\n🔮 开始预测...")
        start_time = time.time()
        
        n_samples = test_images.shape[0]
        print(f"[INFO] 预测样本数: {n_samples}")
        
        # 步骤1: 预处理测试图像
        print("\n📊 步骤 1/2: 预处理测试图像")
        images_uint8 = test_images.astype(np.uint8)
        
        # 步骤2: LBPH分类
        print("\n🎯 步骤 2/2: LBPH分类")
        predicted_labels = []
        
        with tqdm(total=n_samples, desc="分类预测", unit="张") as pbar:
            for img in images_uint8:
                label, _ = self.recognizer.predict(img)
                predicted_labels.append(label if label != -1 else 0)  # 处理未知
                pbar.update(1)
        
        self.prediction_time = time.time() - start_time
        
        print(f"\n✅ 预测完成!")
        print(f"⏱️  预测时间: {self.prediction_time:.2f}秒")
        print(f"⚡  平均预测速度: {n_samples/self.prediction_time:.2f} 张/秒")
        
        return np.array(predicted_labels)
    
    def get_model_info(self):
        """获取模型信息"""
        return {
            'radius': self.radius,
            'neighbors': self.neighbors,
            'grid_x': self.grid_x,
            'grid_y': self.grid_y,
            'threshold': self.threshold,
            'training_samples': len(self.train_labels) if self.train_labels is not None else 0,
            'training_time': self.training_time,
            'prediction_time': self.prediction_time
        }