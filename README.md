# 人脸识别系统

基于多种算法的人脸识别系统，包含ResNet、EigenFace、FisherFace和LBPH四种不同的人脸识别方法。

## 功能特性

### 支持的算法
- **ResNet**: 基于深度残差网络的人脸识别
- **EigenFace**: 基于PCA的经典人脸识别方法
- **FisherFace**: 基于LDA的改进人脸识别方法
- **LBPH**: 基于局部二值模式直方图的人脸识别

### 核心功能
- 数据加载和预处理
- 多种模型训练和验证
- 性能评估和对比
- 结果可视化
- 混淆矩阵生成
- 训练报告导出

## 环境要求

- Python >= 3.7
- CUDA支持（可选，用于ResNet训练加速）

## 安装

1. 克隆项目

git clone <repository-url>
cd face-recognition-system


2. 安装依赖

pip install -r requirements.txt


## 数据集准备

将数据集放置在以下结构中：

data/CelebDataProcessed/
├── person1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── person2/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── ...
```

# 使用方法

### 快速开始

# 运行主程序（训练所有模型）
python main.py

# 仅训练特定模型
python train.py --model resnet
python train.py --model eigenface
python train.py --model fisherface
python train.py --model lbph

# 验证已训练的模型
python validate.py --model resnet
python validate.py --model all
```

### 配置参数

在 `config.py` 中修改配置：
```python
# 数据集路径
DATA_PATH = "data/CelebDataProcessed"

# 训练参数
TRAIN_CONFIG = {
    'test_size': 0.3,
    'random_state': 42,
    'image_size': (128, 128)
}

# 模型参数
MODEL_CONFIG = {
    'resnet': {
        'epochs': 150,
        'batch_size': 32,
        'learning_rate': 0.001
    },
    # ...
}
```

## 性能指标

系统评估以下性能指标：
- **识别准确率**: 正确识别的样本比例
- **精确率**: 预测为正的样本中实际为正的比例
- **召回率**: 实际为正的样本中被正确预测的比例
- **F1分数**: 精确率和召回率的调和平均
- **误报率**: 假正例率

## 输出结果

### 训练结果
- 模型文件保存在 `results/models/`
- 训练日志保存在 `results/reports/`

### 可视化
- 混淆矩阵: `results/plots/confusion_matrix_[model].png`
- 性能对比图: `results/plots/performance_comparison.png`
- PCA/LDA分析图: `results/plots/[model]_analysis.png`

### 评估报告
- 详细评估报告: `results/reports/[model]_validation_results.txt`
- 综合对比报告: `results/reports/comparison_report.txt`

## 项目特色

1. **模块化设计**: 每个算法独立实现，便于扩展和维护
2. **统一接口**: 所有模型使用相同的训练和预测接口
3. **数据增强**: 支持多种数据增强技术提升模型性能
4. **性能优化**: 支持GPU加速、混合精度训练等
5. **可视化丰富**: 提供多种可视化分析工具
6. **结果保存**: 自动保存训练结果和评估报告

## 技术栈

- **深度学习**: PyTorch, torchvision
- **机器学习**: scikit-learn
- **计算机视觉**: OpenCV
- **数据处理**: NumPy, pandas
- **可视化**: Matplotlib, seaborn
- **进度显示**: tqdm

## 贡献

欢迎提交Issue和Pull Request来改进项目。

## 许可证
MIT License
