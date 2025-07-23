"""
模型验证脚本
"""

import argparse
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

from config import *
from utils.data_loader import load_dataset_with_split
from utils.evaluation import comprehensive_evaluation, save_results
from utils.visualization import plot_pca_analysis, plot_lda_analysis, plot_performance_metrics, print_quantitative_comparison

def load_model(model_name):
    """加载已训练的模型"""
    model_path = get_model_path(model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    print(f"📂 已加载模型: {model_path}")
    return model

def validate_single_model(model_name, val_images, val_labels, label_map):
    """验证单个模型"""
    print(f"\n{'='*80}")
    print(f"🔍 验证 {MODEL_MAPPING[model_name]} 模型")
    print('='*80)
    
    try:
        # 加载模型
        model = load_model(model_name)
        
        # 数据类型处理（与训练时保持一致）
        if model_name == 'resnet':
            # ResNet需要彩色图像
            if len(val_images.shape) == 3:  # 灰度图像，需要转换
                import cv2
                val_images_color = []
                for img in val_images:
                    color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    val_images_color.append(color_img)
                val_images = np.array(val_images_color)
        else:
            # 其他模型需要灰度图像
            if len(val_images.shape) == 4:  # 彩色图像，需要转换
                import cv2
                val_images_gray = []
                for img in val_images:
                    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    val_images_gray.append(gray_img)
                val_images = np.array(val_images_gray)
        
        # 评估模型
        results = comprehensive_evaluation(model, val_images, val_labels, label_map, MODEL_MAPPING[model_name])
        
        # 绘制性能指标
        if EVALUATION_CONFIG['save_results']:
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
        if EVALUATION_CONFIG['save_results']:
            save_results(results, MODEL_MAPPING[model_name])
        
        print(f"\n✅ {MODEL_MAPPING[model_name]} 模型验证完成!")
        
        return {model_name: results}
        
    except FileNotFoundError as e:
        print(f"❌ {MODEL_MAPPING[model_name]} 模型文件未找到: {e}")
        print(f"💡 请先运行训练程序或检查模型路径")
        return {}
    except Exception as e:
        print(f"❌ {MODEL_MAPPING[model_name]} 模型验证失败: {e}")
        import traceback
        traceback.print_exc()
        return {}

def validate_models(model_names, val_images, val_labels, label_map):
    """验证多个模型"""
    all_results = {}
    
    for model_name in model_names:
        if model_name in SUPPORTED_MODELS:
            results = validate_single_model(model_name, val_images, val_labels, label_map)
            all_results.update(results)
        else:
            print(f"⚠️ 跳过不支持的模型: {model_name}")
    
    return all_results

def check_available_models():
    """检查可用的已训练模型"""
    available_models = []
    for model_name in SUPPORTED_MODELS:
        model_path = get_model_path(model_name)
        if os.path.exists(model_path):
            available_models.append(model_name)
    
    return available_models

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='验证人脸识别模型')
    parser.add_argument('--model', type=str, choices=SUPPORTED_MODELS + ['all'],
                       default='all', help='要验证的模型')
    parser.add_argument('--data_path', type=str, default=DATA_PATH,
                       help='数据集路径')
    parser.add_argument('--list_models', action='store_true',
                       help='列出可用的已训练模型')
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_arguments()
    
    print("🔍 人脸识别模型验证程序")
    print("="*60)
    
    # 检查可用模型
    available_models = check_available_models()
    
    if args.list_models:
        print(f"\n📋 可用的已训练模型:")
        if available_models:
            for model_name in available_models:
                print(f"   ✅ {MODEL_MAPPING[model_name]} ({model_name})")
        else:
            print("   ❌ 没有找到已训练的模型")
            print("   💡 请先运行训练程序")
        return
    
    if not available_models:
        print(f"\n❌ 没有找到已训练的模型")
        print(f"💡 请先运行训练程序: python train.py")
        return
    
    try:
        # 确定要验证的模型
        if args.model == 'all':
            models_to_validate = available_models
        else:
            if args.model in available_models:
                models_to_validate = [args.model]
            else:
                print(f"❌ 模型 {args.model} 未找到")
                print(f"💡 可用模型: {', '.join(available_models)}")
                return
        
        print(f"\n📋 验证配置:")
        print(f"   模型: {', '.join(models_to_validate)}")
        print(f"   数据路径: {args.data_path}")
        
        # 加载数据集（只需要验证集）
        print(f"\n📊 加载数据集...")
        data = load_dataset_with_split(
            root_dir=args.data_path,
            image_size=DATA_CONFIG['image_size'],
            test_size=DATA_CONFIG['test_size'],
            random_state=DATA_CONFIG['random_state'],
            data_augmentation=False  # 验证时不需要数据增强
        )
        
        train_images, train_labels, val_images, val_labels, label_map = data
        
        print(f"\n✅ 数据集加载完成!")
        print(f"   验证集: {len(val_images)} 张图像")
        print(f"   类别数: {len(label_map)}")
        
        # 验证模型
        all_results = validate_models(models_to_validate, val_images, val_labels, label_map)
        
        print(f"\n🎉 所有模型验证完成!")
        print(f"📁 结果保存在: {RESULTS_DIR}/")
        
        # 显示验证结果摘要
        if all_results:
            print(f"\n📊 验证结果摘要:")
            print("-" * 60)
            print(f"{'模型':<15} {'准确率':<10} {'F1分数':<10}")
            print("-" * 60)
            for model_name, results in all_results.items():
                print(f"{MODEL_MAPPING[model_name]:<15} {results['accuracy']:<10.4f} {results['f1_score']:<10.4f}")
            print("-" * 60)
        
    except FileNotFoundError as e:
        print(f"❌ 数据集错误: {e}")
        print("💡 请确保数据集路径正确，格式为：data/CelebDataProcessed/")
    except Exception as e:
        print(f"❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()