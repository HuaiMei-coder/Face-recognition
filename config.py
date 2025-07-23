"""
配置文件 - 人脸识别系统的所有配置参数
"""

import os

# 基础路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = "data/CelebDataProcessed"
RESULTS_DIR = "results"
MODELS_DIR = os.path.join(RESULTS_DIR, "models")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
REPORTS_DIR = os.path.join(RESULTS_DIR, "reports")

# 创建结果目录
for dir_path in [RESULTS_DIR, MODELS_DIR, PLOTS_DIR, REPORTS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# 数据集配置
DATA_CONFIG = {
    'image_size': (128, 128),
    'test_size': 0.3,
    'random_state': 42,
    'supported_formats': ('.jpg', '.jpeg', '.png', '.bmp')
}

# ResNet模型配置
RESNET_CONFIG = {
    'epochs': 150,
    'batch_size': 32,
    'learning_rate': 0.001,
    'patience': 10,  # 早停耐心值
    'scheduler_patience': 5,  # 学习率调度器耐心值
    'model_name': 'resnet18',
    'pretrained': True
}

# EigenFace模型配置
EIGENFACE_CONFIG = {
    'n_components': None,  # 自动确定
    'variance_ratio': 0.98,
    'knn_neighbors': 5
}

# FisherFace模型配置
FISHERFACE_CONFIG = {
    'n_components': None,  # 自动确定
    'variance_ratio': 0.98,
    'knn_neighbors': 5
}

# LBPH模型配置
LBPH_CONFIG = {
    'radius': 1,
    'neighbors': 8,
    'grid_x': 8,
    'grid_y': 8,
    'threshold': 100.0
}

# 训练配置
TRAIN_CONFIG = {
    'data_augmentation': True,
    'use_weighted_sampling': True,
    'save_models': True,
    'save_plots': True,
    'save_reports': True
}

# 评估配置
EVALUATION_CONFIG = {
    'metrics': ['accuracy', 'precision', 'recall', 'f1_score', 'false_positive_rate'],
    'show_confusion_matrix': True,
    'show_examples': True,
    'save_results': True
}

# 可视化配置
VISUALIZATION_CONFIG = {
    'figure_size': (12, 10),
    'dpi': 100,
    'style': 'seaborn',
    'color_palette': 'Blues',
    'font_size': 12
}

# 模型名称映射
MODEL_MAPPING = {
    'resnet': 'ResNet',
    'eigenface': 'EigenFace', 
    'fisherface': 'FisherFace',
    'lbph': 'LBPH'
}

# 所有支持的模型
SUPPORTED_MODELS = list(MODEL_MAPPING.keys())

# 获取模型配置的函数
def get_model_config(model_name):
    """获取指定模型的配置"""
    config_map = {
        'resnet': RESNET_CONFIG,
        'eigenface': EIGENFACE_CONFIG,
        'fisherface': FISHERFACE_CONFIG,
        'lbph': LBPH_CONFIG
    }
    return config_map.get(model_name.lower(), {})

# 文件路径生成函数
def get_model_path(model_name):
    """获取模型保存路径"""
    return os.path.join(MODELS_DIR, f"{model_name}_model.pkl")

def get_plot_path(plot_name):
    """获取图表保存路径"""
    return os.path.join(PLOTS_DIR, f"{plot_name}.png")

def get_report_path(report_name):
    """获取报告保存路径"""
    return os.path.join(REPORTS_DIR, f"{report_name}.txt")