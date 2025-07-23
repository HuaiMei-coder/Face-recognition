"""
FisherFace人脸识别模型实现
"""

import numpy as np
import time
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier

class EnhancedFisherFaceRecognizer:
    """
    增强版FisherFace识别器
    """
    def __init__(self, n_components=None, variance_ratio=0.98, knn_neighbors=5):
        """
        初始化FisherFace识别器
        """
        self.n_components = n_components
        self.variance_ratio = variance_ratio
        self.pca = None
        self.lda = None
        self.mean_face = None
        self.projected_faces = None
        self.train_labels = None
        self.training_time = None
        self.prediction_time = None
        self.knn = KNeighborsClassifier(n_neighbors=knn_neighbors, weights='distance')
        
    def train(self, train_images, train_labels):
        """
        训练FisherFace模型
        """
        print("[INFO] 🚀 开始训练FisherFace模型...")
        start_time = time.time()
        
        # 步骤1: 数据预处理
        print("\n📊 步骤 1/5: 数据预处理")
        n_samples = train_images.shape[0]
        n_classes = len(np.unique(train_labels))
        
        print(f"[INFO] 训练样本数: {n_samples}")
        print(f"[INFO] 类别数: {n_classes}")
        print(f"[INFO] 图像尺寸: {train_images.shape[1:]} -> 特征维度: {train_images.shape[1] * train_images.shape[2]}")
        
        # 将图像展平为向量
        print("[INFO] 展平图像数据...")
        flattened_images = []
        for i in tqdm(range(n_samples), desc="展平图像", unit="张"):
            flattened = train_images[i].reshape(-1).astype(np.float64)
            flattened_images.append(flattened)
        flattened_images = np.array(flattened_images)
        
        # 步骤2: 计算平均脸
        print("\n👤 步骤 2/5: 计算平均脸")
        print("[INFO] 计算所有训练图像的平均脸...")
        self.mean_face = np.mean(flattened_images, axis=0)
        print("✅ 平均脸计算完成")
        
        # 步骤3: 数据中心化
        print("\n🎯 步骤 3/5: 数据中心化")
        print("[INFO] 从每张图像中减去平均脸...")
        centered_data = []
        for i in tqdm(range(len(flattened_images)), desc="中心化处理", unit="张"):
            centered = flattened_images[i] - self.mean_face
            centered_data.append(centered)
        centered_data = np.array(centered_data)
        print("✅ 数据中心化完成")
        
        # 步骤4: 确定PCA组件数
        print("\n🔧 步骤 4/5: 确定PCA组件数")
        if self.n_components is None:
            print("[INFO] 自动确定最优组件数...")
            print("[INFO] 执行初始PCA分析以确定组件数...")
            
            # 使用较小的样本进行初始分析以节省时间
            sample_size = min(1000, len(centered_data))
            sample_indices = np.random.choice(len(centered_data), sample_size, replace=False)
            sample_data = centered_data[sample_indices]
            
            temp_pca = PCA()
            with tqdm(total=100, desc="PCA分析", unit="%") as pbar:
                temp_pca.fit(sample_data)
                pbar.update(100)
            
            cumsum_ratio = np.cumsum(temp_pca.explained_variance_ratio_)
            self.n_components = np.argmax(cumsum_ratio >= self.variance_ratio) + 1
            self.n_components = min(self.n_components, n_samples - 1, 200)  # 限制最大组件数
            
            print(f"[INFO] 基于{self.variance_ratio*100:.1f}%方差保留确定组件数: {self.n_components}")
        else:
            print(f"[INFO] 使用预设组件数: {self.n_components}")
        
        # 为LDA调整PCA组件数
        self.n_components = min(self.n_components, n_samples - n_classes)
        print(f"[INFO] 为LDA调整PCA组件数: {self.n_components}")
        
        # 步骤5: 应用PCA降维
        print("\n🎨 步骤 5/5: PCA降维和特征投影")
        print(f"[INFO] 应用PCA降维: {flattened_images.shape[1]} -> {self.n_components} 维")
        
        self.pca = PCA(n_components=self.n_components, whiten=True)
        
        # 使用进度条显示PCA训练过程
        with tqdm(total=100, desc="PCA训练", unit="%") as pbar:
            pbar.set_description("计算协方差矩阵")
            pbar.update(20)
            
            pbar.set_description("特征值分解")
            pbar.update(30)
            
            # 实际PCA训练
            pca_projected = self.pca.fit_transform(centered_data)
            pbar.set_description("投影训练数据")
            pbar.update(50)
            
            pbar.set_description("PCA训练完成")
            pbar.update(100)
        
        # 应用LDA
        print("\n[INFO] 应用LDA降维...")
        self.lda = LinearDiscriminantAnalysis()
        with tqdm(total=100, desc="LDA训练", unit="%") as pbar:
            self.projected_faces = self.lda.fit_transform(pca_projected, train_labels)
            pbar.update(100)
        
        self.train_labels = train_labels
        self.training_time = time.time() - start_time
        
        # 训练k-NN分类器
        print("\n[INFO] 训练k-NN分类器...")
        self.knn.fit(self.projected_faces, self.train_labels)
        
        # 输出训练结果
        explained_variance = sum(self.lda.explained_variance_ratio_)
        print(f"\n✅ 训练完成!")
        print(f"📈 LDA解释方差比例: {explained_variance:.4f} ({explained_variance*100:.2f}%)")
        print(f"⏱️  总训练时间: {self.training_time:.2f}秒")
        print(f"🎯 投影后特征维度: {self.projected_faces.shape[1]}")
        print(f"💾 训练数据投影形状: {self.projected_faces.shape}")
        
    def predict(self, test_images):
        """
        预测测试图像的标签
        """
        if self.pca is None or self.lda is None:
            raise ValueError("模型必须先训练才能进行预测")
        
        print("\n🔮 开始预测...")
        start_time = time.time()
        
        n_samples = test_images.shape[0]
        print(f"[INFO] 预测样本数: {n_samples}")
        
        # 步骤1: 预处理测试图像
        print("\n📊 步骤 1/3: 预处理测试图像")
        flattened_test = []
        for i in tqdm(range(n_samples), desc="展平测试图像", unit="张"):
            flattened = test_images[i].reshape(-1).astype(np.float64)
            flattened_test.append(flattened)
        flattened_test = np.array(flattened_test)
        
        # 步骤2: 中心化并投影
        print("\n🎯 步骤 2/3: 中心化并投影到特征空间")
        centered_test = []
        for i in tqdm(range(len(flattened_test)), desc="中心化处理", unit="张"):
            centered = flattened_test[i] - self.mean_face
            centered_test.append(centered)
        centered_test = np.array(centered_test)
        
        # 投影到特征空间
        print("[INFO] 投影到PCA特征空间...")
        with tqdm(total=100, desc="PCA投影", unit="%") as pbar:
            projected_test = self.pca.transform(centered_test)
            pbar.update(100)
        
        print("[INFO] 投影到LDA特征空间...")
        with tqdm(total=100, desc="LDA投影", unit="%") as pbar:
            projected_test = self.lda.transform(projected_test)
            pbar.update(100)
        
        # 步骤3: k-NN分类
        print("\n🎯 步骤 3/3: k-NN分类")
        
        with tqdm(total=len(projected_test), desc="分类预测", unit="张") as pbar:
            predicted_labels = self.knn.predict(projected_test)
            pbar.update(len(projected_test))
        
        self.prediction_time = time.time() - start_time
        
        print(f"\n✅ 预测完成!")
        print(f"⏱️  预测时间: {self.prediction_time:.2f}秒")
        print(f"⚡  平均预测速度: {n_samples/self.prediction_time:.2f} 张/秒")
        
        return np.array(predicted_labels)
    
    def get_model_info(self):
        """获取模型信息"""
        return {
            'n_components': self.n_components,
            'training_samples': len(self.train_labels) if self.train_labels is not None else 0,
            'training_time': self.training_time,
            'prediction_time': self.prediction_time,
            'explained_variance': sum(self.lda.explained_variance_ratio_) if self.lda else 0
        }