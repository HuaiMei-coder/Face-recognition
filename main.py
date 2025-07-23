"""
主程序入口 - 人脸识别系统
"""

import argparse
import time
import sys
import warnings
warnings.filterwarnings('ignore')

from config import *
from utils.data_loader import load_dataset_with_split
from utils.evaluation import comprehensive_comparison
from train import train_models
from validate import validate_models

def print_banner():
    """打印程序横幅"""
    print("="*80)
    print("🎯 人脸识别系统 - 多算法对比平台")
    print("="*80)
    print("支持的算法:")
    print("  • ResNet    - 基于深度残差网络的人脸识别")
    print("  • EigenFace - 基于PCA的经典人脸识别方法") 
    print("  • FisherFace- 基于LDA的改进人脸识别方法")
    print("  • LBPH      - 基于局部二值模式直方图的人脸识别")
    print("="*80)

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='人脸识别系统')
    parser.add_argument('--mode', type=str, choices=['train', 'validate', 'full'],
                       default='full', help='运行模式: train(训练), validate(验证), full(完整流程)')
    parser.add_argument('--models', type=str, nargs='+', 
                       choices=SUPPORTED_MODELS + ['all'],
                       default=['all'], help='要运行的模型')
    parser.add_argument('--data_path', type=str, default=DATA_PATH,
                       help='数据集路径')
    parser.add_argument('--no_comparison', action='store_true',
                       help='跳过模型对比')
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_arguments()
    print_banner()
    
    try:
        start_time = time.time()
        
        # 确定要运行的模型
        if 'all' in args.models:
            models_to_run = SUPPORTED_MODELS
        else:
            models_to_run = args.models
        
        print(f"\n📋 运行配置:")
        print(f"   模式: {args.mode}")
        print(f"   模型: {', '.join(models_to_run)}")
        print(f"   数据路径: {args.data_path}")
        
        # 检查数据集
        print(f"\n📂 检查数据集...")
        if not os.path.exists(args.data_path):
            raise FileNotFoundError(f"数据集路径不存在: {args.data_path}")
        
        # 加载数据集
        print(f"\n📊 加载数据集...")
        data = load_dataset_with_split(
            root_dir=args.data_path,
            image_size=DATA_CONFIG['image_size'],
            test_size=DATA_CONFIG['test_size'],
            random_state=DATA_CONFIG['random_state']
        )
        
        train_images, train_labels, val_images, val_labels, label_map = data
        
        print(f"\n✅ 数据集加载完成!")
        print(f"   训练集: {len(train_images)} 张图像")
        print(f"   验证集: {len(val_images)} 张图像") 
        print(f"   类别数: {len(label_map)}")
        
        # 执行相应模式
        results = {}
        
        if args.mode in ['train', 'full']:
            print(f"\n🚀 开始训练模式...")
            results.update(train_models(
                models_to_run, 
                train_images, train_labels, 
                val_images, val_labels, 
                label_map
            ))
        
        if args.mode in ['validate', 'full']:
            print(f"\n🔍 开始验证模式...")
            validation_results = validate_models(
                models_to_run,
                val_images, val_labels,
                label_map
            )
            results.update(validation_results)
        
        # 模型对比
        if not args.no_comparison and len(models_to_run) > 1 and results:
            print(f"\n📊 开始模型对比分析...")
            comprehensive_comparison(results, models_to_run)
        
        # 运行完成
        total_time = time.time() - start_time
        print(f"\n🎉 所有任务完成!")
        print(f"⏱️  总运行时间: {total_time/60:.2f} 分钟")
        print(f"📁 结果保存在: {RESULTS_DIR}/")
        
    except FileNotFoundError as e:
        print(f"❌ 数据集错误: {e}")
        print("💡 请确保数据集路径正确，格式为：data/CelebDataProcessed/")
        sys.exit(1)
    except KeyboardInterrupt:
        print(f"\n⚠️  用户中断程序运行")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()