"""
模型训练脚本
"""

import argparse
import pickle
import time
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from config import *
from models import ResNetFaceRecognizer, EnhancedEigenFaceRecognizer, EnhancedFisherFaceRecognizer, EnhancedLBPHRecognizer
from utils.data_loader import load_dataset_with_split
from utils.evaluation import comprehensive_evaluation, save_results
from utils.visualization import plot_pca_analysis, plot_lda_analysis, plot_performance_metrics, print_quantitative_comparison

def create_model(model_name, num_classes):
    """创建指定的模型"""
    if model_name == 'resnet':
        config = get_model_config('resnet')
        return ResNetFaceRecognizer(
            num_classes=num_classes,
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            learning_rate=config['learning_rate']
        )
    elif model_name == 'eigenface':
        config = get_model_config('eigenface')
        return EnhancedEigenFaceRecognizer(
            n_components=config['n_components'],
            variance_ratio=config['variance_ratio'],
            knn_neighbors=config['knn_neighbors']
        )
    elif model_name == 'fisherface':
        config = get_model_config('fisherface')
        return EnhancedFisherFaceRecognizer(
            n_components=config['n_components'],
            variance_ratio=config['variance_ratio'],
            knn_neighbors=config['knn_neighbors']
        )
    elif model_name == 'lbph':
        config = get_model_config('lbph')
        return EnhancedLBPHRecognizer(
            radius=config['radius'],
            neighbors=config['neighbors'],
            grid_x=config['grid_x'],
            grid_y=config['grid_y'],
            threshold=config['threshold']
        )
    else:
        raise ValueError(f"不支持的模型: {model_name}")

def save_model(model, model_name):
    """保存模型"""
    if model_name == 'lbph':
        # LBPH模型使用OpenCV自带的保存方法
        model_path = get_model_path(model_name).replace('.pkl', '.yml')
        model.recognizer.save(model_path)  # 假设model有recognizer属性
        print(f"💾 LBPH模型已保存到: {model_path}")
    else:
        # 其他模型使用pickle保存
        model_path = get_model_path(model_name)
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"💾 模型已保存到: {model_path}")

def train_single_model(model_name, train_images, train_labels, val_images, val_labels, label_map):
    """训练单个模型"""
    print(f"\n{'='*80}")
    print(f"🚀 开始训练 {MODEL_MAPPING[model_name]} 模型")
    print('='*80)
    
    # 确定数据类型（ResNet需要彩色图像，其他需要灰度图像）
    if model_name == 'resnet':
        # ResNet需要彩色图像，确保数据是彩色的
        if len(train_images.shape) == 3:  # 灰度图像，需要转换
            import cv2
            train_images_color = []
            val_images_color = []
            for img in train_images:
                color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                train_images_color.append(color_img)
            for img in val_images:
                color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                val_images_color.append(color_img)
            train_images = np.array(train_images_color)
            val_images = np.array(val_images_color)
    else:
        # 其他模型需要灰度图像
        if len(train_images.shape) == 4:  # 彩色图像，需要转换
            import cv2
            train_images_gray = []
            val_images_gray = []
            for img in train_images:
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                train_images_gray.append(gray_img)
            for img in val_images:
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                val_images_gray.append(gray_img)
            train_images = np.array(train_images_gray)
            val_images = np.array(val_images_gray)
    
    try:
        # 创建模型
        num_classes = len(np.unique(train_labels))
        model = create_model(model_name, num_classes)
        
        # 训练模型
        start_time = time.time()
        if model_name == 'resnet':
            model.train_model(train_images, train_labels, val_images, val_labels)
        else:
            model.train(train_images, train_labels)
        
        training_time = time.time() - start_time
        
        # 保存模型
        if TRAIN_CONFIG['save_models']:
            save_model(model, model_name)
        
        # 评估模型
        print(f"\n🔍 评估 {MODEL_MAPPING[model_name]} 模型性能...")
        results = comprehensive_evaluation(model, val_images, val_labels, label_map, MODEL_MAPPING[model_name])
        
        # 绘制性能指标
        if TRAIN_CONFIG['save_plots']:
            plot_performance_metrics(results, MODEL_MAPPING[model_name])
            
            # 绘制特殊分析图
            if model_name == 'eigenface':
                plot_pca_analysis(model, MODEL_MAPPING[model_name])
            elif model_name == 'fisherface':
                plot_pca_analysis(model, MODEL_MAPPING[model_name])
                plot_lda_analysis(model, MODEL_MAPPING[model_name])
        
        # 打印定量结果
        print_quantitative_comparison(results, MODEL_MAPPING[model_name])
        
        # 保存结果
        if TRAIN_CONFIG['save_reports']:
            save_results(results, MODEL_MAPPING[model_name])
        
        print(f"\n✅ {MODEL_MAPPING[model_name]} 模型训练完成!")
        print(f"⏱️  总训练时间: {training_time/60:.2f} 分钟")
        
        return {model_name: results}
        
    except Exception as e:
        print(f"❌ {MODEL_MAPPING[model_name]} 模型训练失败: {e}")
        import traceback
        traceback.print_exc()
        return {}

def train_models(model_names, train_images, train_labels, val_images, val_labels, label_map):
    """训练多个模型"""
    all_results = {}
    
    for model_name in model_names:
        if model_name in SUPPORTED_MODELS:
            results = train_single_model(model_name, train_images, train_labels, val_images, val_labels, label_map)
            all_results.update(results)
        else:
            print(f"⚠️ 跳过不支持的模型: {model_name}")
    
    return all_results

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='训练人脸识别模型')
    parser.add_argument('--model', type=str, choices=SUPPORTED_MODELS + ['all'],
                       default='all', help='要训练的模型')
    parser.add_argument('--data_path', type=str, default=DATA_PATH,
                       help='数据集路径')
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_arguments()
    
    print("🎯 人脸识别模型训练程序")
    print("="*60)
    
    try:
        # 确定要训练的模型
        if args.model == 'all':
            models_to_train = SUPPORTED_MODELS
        else:
            models_to_train = [args.model]
        
        print(f"\n📋 训练配置:")
        print(f"   模型: {', '.join(models_to_train)}")
        print(f"   数据路径: {args.data_path}")
        
        # 加载数据集
        print(f"\n📊 加载数据集...")
        data = load_dataset_with_split(
            root_dir=args.data_path,
            image_size=DATA_CONFIG['image_size'],
            test_size=DATA_CONFIG['test_size'],
            random_state=DATA_CONFIG['random_state'],
            data_augmentation=TRAIN_CONFIG['data_augmentation']
        )
        
        train_images, train_labels, val_images, val_labels, label_map = data
        
        print(f"\n✅ 数据集加载完成!")
        print(f"   训练集: {len(train_images)} 张图像")
        print(f"   验证集: {len(val_images)} 张图像")
        print(f"   类别数: {len(label_map)}")
        
        # 训练模型
        start_time = time.time()
        all_results = train_models(models_to_train, train_images, train_labels, val_images, val_labels, label_map)
        total_time = time.time() - start_time
        
        print(f"\n🎉 所有模型训练完成!")
        print(f"⏱️  总训练时间: {total_time/60:.2f} 分钟")
        print(f"📁 结果保存在: {RESULTS_DIR}/")
        
    except FileNotFoundError as e:
        print(f"❌ 数据集错误: {e}")
        print("💡 请确保数据集路径正确，格式为：data/CelebDataProcessed/")
    except Exception as e:
        print(f"❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()