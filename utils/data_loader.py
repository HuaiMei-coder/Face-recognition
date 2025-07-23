"""
数据加载和预处理工具
"""

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from torch.utils.data import WeightedRandomSampler
import torch

def align_face(image):
    """
    对图像进行人脸对齐，确保眼睛水平对齐
    """
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    faces = face_cascade.detectMultiScale(image, 1.3, 5)
    if len(faces) == 0:
        return image  # 无脸部检测，返回原始
    
    # 取第一个脸部
    (x, y, w, h) = faces[0]
    face = image[y:y+h, x:x+w]
    
    eyes = eye_cascade.detectMultiScale(face)
    if len(eyes) < 2:
        return image  # 少于两个眼睛，返回原始
    
    # 取前两个眼睛
    eye1 = eyes[0]
    eye2 = eyes[1]
    
    # 计算眼睛中心
    eye1_center = (eye1[0] + eye1[2]//2, eye1[1] + eye1[3]//2)
    eye2_center = (eye2[0] + eye2[2]//2, eye2[1] + eye2[3]//2)
    
    # 计算旋转角度
    dx = eye2_center[0] - eye1_center[0]
    dy = eye2_center[1] - eye1_center[1]
    angle = np.degrees(np.arctan2(dy, dx)) - 90 if eye1_center[0] > eye2_center[0] else np.degrees(np.arctan2(dy, dx))
    
    # 获取旋转矩阵
    (h_img, w_img) = image.shape[:2]
    center = (w_img // 2, h_img // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # 旋转图像
    aligned = cv2.warpAffine(image, M, (w_img, h_img), flags=cv2.INTER_CUBIC)
    
    return aligned

def augment_image(image):
    """
    对单张图像进行增强数据增强
    """
    augmented_images = [image]
    
    # 水平翻转
    flipped = cv2.flip(image, 1)
    augmented_images.append(flipped)
    
    # 随机旋转（±20度）
    angle = np.random.uniform(-20, 20)
    if len(image.shape) == 3:
        h, w, _ = image.shape
    else:
        h, w = image.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    augmented_images.append(rotated)
    
    # 颜色抖动（调整亮度和对比度）
    brightness = np.random.uniform(0.7, 1.3)
    contrast = np.random.uniform(0.7, 1.3)
    jittered = np.clip(image * brightness * contrast, 0, 255).astype(np.uint8)
    augmented_images.append(jittered)
    
    # 随机缩放（0.8-1.2）
    scale = np.random.uniform(0.8, 1.2)
    new_size = (int(w * scale), int(h * scale))
    scaled = cv2.resize(image, new_size)
    scaled = cv2.resize(scaled, (w, h))  # 缩回原大小
    augmented_images.append(scaled)
    
    # 添加高斯噪声
    noise = np.random.normal(0, np.random.uniform(5, 15), image.shape)
    noisy = np.clip(image + noise, 0, 255).astype(np.uint8)
    augmented_images.append(noisy)
    
    # 高斯模糊
    blur_kernel = np.random.choice([3, 5])
    blurred = cv2.GaussianBlur(image, (blur_kernel, blur_kernel), 0)
    augmented_images.append(blurred)
    
    return augmented_images

def load_dataset_with_split(root_dir, image_size=(128, 128), test_size=0.3, random_state=42, 
                           color_mode='auto', data_augmentation=True):
    """
    加载数据集并按70%-30%划分训练集和验证集
    
    Parameters:
        root_dir (str): 数据集根目录
        image_size (tuple): 图像调整尺寸
        test_size (float): 验证集比例
        random_state (int): 随机种子
        color_mode (str): 颜色模式 - 'auto', 'color', 'gray'
        data_augmentation (bool): 是否进行数据增强
        
    Returns:
        tuple: (train_images, train_labels, val_images, val_labels, label_map)
    """
    print(f"[INFO] 从 {root_dir} 加载数据集...")
    
    if not os.path.exists(root_dir):
        raise FileNotFoundError(f"数据集目录不存在: {root_dir}")
    
    # 获取所有人物文件夹
    people = sorted([p for p in os.listdir(root_dir) 
                    if os.path.isdir(os.path.join(root_dir, p))])
    
    print(f"[INFO] 发现 {len(people)} 个人物类别")
    
    # 创建标签编码器
    le = LabelEncoder()
    le.fit(people)
    label_map = dict(zip(le.transform(people), people))
    
    all_images = []
    all_labels = []
    
    # 支持的图像格式
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp')
    
    # 加载所有图像
    for person in tqdm(people, desc="加载图像"):
        person_path = os.path.join(root_dir, person)
        person_images = []
        
        for fname in os.listdir(person_path):
            if fname.lower().endswith(supported_formats):
                img_path = os.path.join(person_path, fname)
                try:
                    # 根据color_mode决定读取方式
                    if color_mode == 'gray':
                        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    elif color_mode == 'color':
                        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                    else:  # auto模式，根据第一张图像决定
                        if not all_images:  # 第一张图像
                            test_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                            if test_img is not None and len(test_img.shape) == 3:
                                color_mode = 'color'
                                img = test_img
                            else:
                                color_mode = 'gray'
                                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        else:
                            if color_mode == 'color':
                                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                            else:
                                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    
                    if img is not None:
                        img = cv2.resize(img, image_size)
                        # 应用人脸对齐
                        if color_mode == 'gray':
                            img = align_face(img)
                            # 应用直方图均衡化
                            img = cv2.equalizeHist(img)
                        else:
                            # 对彩色图像的每个通道应用对齐
                            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                            aligned_gray = align_face(gray_img)
                            # 如果对齐成功，应用相同的变换到彩色图像
                            img = align_face(img)
                        person_images.append(img)
                except Exception as e:
                    print(f"警告: 无法加载图像 {img_path}: {e}")
        
        if len(person_images) == 0:
            print(f"警告: {person} 没有有效图像")
            continue
        
        # 为每个人的图像分配标签
        person_label = le.transform([person])[0]
        all_images.extend(person_images)
        all_labels.extend([person_label] * len(person_images))
    
    all_images = np.array(all_images)
    all_labels = np.array(all_labels)
    
    print(f"[INFO] 总共加载 {len(all_images)} 张图像")
    print(f"[INFO] 图像形状: {all_images.shape}")
    print(f"[INFO] 图像模式: {'彩色' if color_mode == 'color' else '灰度'}")
    print(f"[INFO] 类别数: {len(np.unique(all_labels))}")
    
    # 按70%-30%划分数据集
    train_images, val_images, train_labels, val_labels = train_test_split(
        all_images, all_labels, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=all_labels  # 保证每个类别在训练集和验证集中都有代表
    )
    
    # 数据增强（仅对训练集）
    if data_augmentation:
        print("\n[INFO] 对训练集进行数据增强...")
        augmented_train_images = []
        augmented_train_labels = []
        
        for img, label in tqdm(zip(train_images, train_labels), total=len(train_images), desc="数据增强"):
            # 对每张图像进行增强
            aug_images = augment_image(img)
            augmented_train_images.extend(aug_images)
            augmented_train_labels.extend([label] * len(aug_images))
        
        train_images = np.array(augmented_train_images)
        train_labels = np.array(augmented_train_labels)
        
        print(f"[INFO] 增强后训练集: {len(train_images)} 张图像")
    
    print(f"[INFO] 训练集: {len(train_images)} 张图像")
    print(f"[INFO] 验证集: {len(val_images)} 张图像")
    
    return train_images, train_labels, val_images, val_labels, label_map

def create_weighted_sampler(train_labels):
    """创建类别平衡采样器"""
    print("\n[INFO] 计算类别平衡采样权重...")
    unique_labels, counts = np.unique(train_labels, return_counts=True)
    class_weights = 1.0 / counts
    sample_weights = np.array([class_weights[np.where(unique_labels == label)[0][0]] for label in train_labels])
    
    # 创建 WeightedRandomSampler
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_labels), replacement=True)
    
    return sampler