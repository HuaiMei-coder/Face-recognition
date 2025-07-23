"""
性能评估工具
"""

import os
import numpy as np
import time
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from config import get_report_path, get_plot_path
from .visualization import plot_confusion_matrix, plot_performance_metrics, show_prediction_examples
from .data_loader import align_face
import cv2

def comprehensive_evaluation(model, val_images, val_labels, label_map, model_name="Model"):
    """
    全面评估模型性能
    """
    print(f"\n{'='*60}")
    print(f"🎯 {model_name} 模型性能评估")
    print('='*60)
    
    # 预处理验证图像
    print("\n[INFO] 预处理验证图像...")
    processed_val_images = []
    for img in tqdm(val_images, desc="处理验证图像"):
        if len(img.shape) == 2:  # 灰度图像
            aligned = align_face(img)
            equalized = cv2.equalizeHist(aligned)
            processed_val_images.append(equalized)
        else:  # 彩色图像
            aligned = align_face(img)
            processed_val_images.append(aligned)
    val_images = np.array(processed_val_images)
    
    # 预测
    print("\n🔮 开始模型评估...")
    evaluation_start = time.time()
    
    predictions = model.predict(val_images)
    
    print(f"\n📊 计算评估指标...")
    # 计算各种指标
    with tqdm(total=100, desc="计算指标", unit="%") as pbar:
        accuracy = accuracy_score(val_labels, predictions)
        pbar.update(25)
        
        precision = precision_score(val_labels, predictions, average='macro', zero_division=0)
        pbar.update(25)
        
        recall = recall_score(val_labels, predictions, average='macro', zero_division=0)
        pbar.update(25)
        
        f1 = f1_score(val_labels, predictions, average='macro', zero_division=0)
        pbar.update(25)
    
    # 计算误报率
    cm = confusion_matrix(val_labels, predictions)
    
    # 对于多分类，计算平均误报率
    fp_rates = []
    for i in range(len(np.unique(val_labels))):
        tp = cm[i, i]  # 真正例
        fp = cm[:, i].sum() - tp  # 假正例
        tn = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + tp  # 真负例
        fn = cm[i, :].sum() - tp  # 假负例
        
        if (fp + tn) > 0:
            fpr = fp / (fp + tn)
            fp_rates.append(fpr)
    
    avg_fpr = np.mean(fp_rates) if fp_rates else 0
    
    evaluation_time = time.time() - evaluation_start
    
    # 输出关键性能指标
    print(f"\n📊 核心性能指标:")
    print(f"✅ 识别准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"🎯 精确率: {precision:.4f} ({precision*100:.2f}%)")
    print(f"📈 召回率: {recall:.4f} ({recall*100:.2f}%)")
    print(f"🏆 F1分数: {f1:.4f} ({f1*100:.2f}%)")
    print(f"⚠️  误报率: {avg_fpr:.4f} ({avg_fpr*100:.2f}%)")
    
    # 绘制混淆矩阵
    print(f"\n🎨 生成混淆矩阵...")
    plot_confusion_matrix(cm, label_map, val_labels, predictions, model_name)
    
    # 分析性能
    analyze_performance(val_labels, predictions, label_map)
    
    # 显示正确和错误识别的示例
    show_prediction_examples(val_images, val_labels, predictions, label_map, model_name)
    
    # 返回结果字典
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'false_positive_rate': avg_fpr,
        'evaluation_time': evaluation_time,
        'model_info': model.get_model_info() if hasattr(model, 'get_model_info') else {}
    }
    
    return results

def analyze_performance(true_labels, predictions, label_map):
    """分析模型性能"""
    print(f"\n🔍 性能分析:")
    
    # 找出最好和最差的类别
    unique_labels = np.unique(true_labels)
    class_accuracies = []
    
    for label in unique_labels:
        mask = true_labels == label
        if np.sum(mask) > 0:
            class_acc = accuracy_score(true_labels[mask], predictions[mask])
            class_accuracies.append((label, class_acc, np.sum(mask)))
    
    # 排序
    class_accuracies.sort(key=lambda x: x[1], reverse=True)
    
    print(f"表现最好的3个类别:")
    for i, (label, acc, count) in enumerate(class_accuracies[:3]):
        name = label_map.get(label, f"Class_{label}")
        print(f"  {i+1}. {name}: {acc:.4f} ({count}个样本)")
    
    print(f"表现最差的3个类别:")
    for i, (label, acc, count) in enumerate(class_accuracies[-3:]):
        name = label_map.get(label, f"Class_{label}")
        print(f"  {i+1}. {name}: {acc:.4f} ({count}个样本)")

def save_results(results, model_name):
    """保存验证结果到文件"""
    filename = get_report_path(f"{model_name.lower()}_validation_results")
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"{model_name} 模型验证结果\n")
        f.write("="*50 + "\n\n")
        
        f.write("核心性能指标:\n")
        f.write(f"识别准确率: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)\n")
        f.write(f"精确率: {results['precision']:.4f} ({results['precision']*100:.2f}%)\n")
        f.write(f"召回率: {results['recall']:.4f} ({results['recall']*100:.2f}%)\n")
        f.write(f"F1分数: {results['f1_score']:.4f} ({results['f1_score']*100:.2f}%)\n")
        f.write(f"误报率: {results['false_positive_rate']:.4f} ({results['false_positive_rate']*100:.2f}%)\n\n")
        
        f.write(f"评估时间: {results['evaluation_time']:.2f}秒\n")
        
        # 添加模型信息
        if 'model_info' in results and results['model_info']:
            f.write("\n模型信息:\n")
            for key, value in results['model_info'].items():
                f.write(f"{key}: {value}\n")
        
        f.write(f"验证时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"📄 结果已保存到: {filename}")

def comprehensive_comparison(all_results, model_names):
    """
    进行综合的模型对比分析
    """
    print(f"\n{'='*80}")
    print("📊 模型综合对比分析")
    print('='*80)
    
    # 整理对比数据
    comparison_data = {}
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'false_positive_rate']
    
    for metric in metrics:
        comparison_data[metric] = {}
        for model_name in model_names:
            if model_name in all_results:
                comparison_data[metric][model_name] = all_results[model_name].get(metric, 0)
    
    # 打印对比表格
    print(f"\n📋 性能对比表:")
    print("-" * 80)
    print(f"{'模型':<12} {'准确率':<10} {'精确率':<10} {'召回率':<10} {'F1分数':<10} {'误报率':<10}")
    print("-" * 80)
    
    for model_name in model_names:
        if model_name in all_results:
            results = all_results[model_name]
            print(f"{model_name:<12} "
                  f"{results['accuracy']:<10.4f} "
                  f"{results['precision']:<10.4f} "
                  f"{results['recall']:<10.4f} "
                  f"{results['f1_score']:<10.4f} "
                  f"{results['false_positive_rate']:<10.4f}")
    
    print("-" * 80)
    
    # 找出最佳模型
    best_models = {}
    for metric in metrics:
        if metric == 'false_positive_rate':  # 误报率越低越好
            best_model = min(comparison_data[metric].items(), key=lambda x: x[1])
        else:  # 其他指标越高越好
            best_model = max(comparison_data[metric].items(), key=lambda x: x[1])
        best_models[metric] = best_model
    
    print(f"\n🏆 各指标最佳模型:")
    for metric, (model_name, value) in best_models.items():
        print(f"  {metric.replace('_', ' ').title()}: {model_name} ({value:.4f})")
    
    # 综合排名
    print(f"\n🥇 综合性能排名:")
    overall_scores = {}
    for model_name in model_names:
        if model_name in all_results:
            results = all_results[model_name]
            # 计算综合得分（简单平均，误报率取反）
            score = (results['accuracy'] + results['precision'] + 
                    results['recall'] + results['f1_score'] - 
                    results['false_positive_rate']) / 4
            overall_scores[model_name] = score
    
    ranked_models = sorted(overall_scores.items(), key=lambda x: x[1], reverse=True)
    for i, (model_name, score) in enumerate(ranked_models, 1):
        print(f"  {i}. {model_name}: {score:.4f}")
    
    # 生成对比图表
    plot_comparison_charts(all_results, model_names)
    
    # 保存对比报告
    save_comparison_report(all_results, model_names, best_models, ranked_models)

def plot_comparison_charts(all_results, model_names):
    """绘制对比图表"""
    import matplotlib.pyplot as plt
    
    # 性能指标对比图
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    for i, metric in enumerate(metrics):
        values = [all_results[model].get(metric, 0) for model in model_names if model in all_results]
        models = [model for model in model_names if model in all_results]
        
        bars = axes[i].bar(models, values)
        axes[i].set_title(f'{metric.replace("_", " ").title()} Comparison')
        axes[i].set_ylim(0, 1)
        axes[i].grid(axis='y', alpha=0.3)
        
        # 添加数值标签
        for bar, value in zip(bars, values):
            axes[i].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(get_plot_path('performance_comparison'))
    plt.show()

def save_comparison_report(all_results, model_names, best_models, ranked_models):
    """保存对比报告"""
    filename = get_report_path('comparison_report')
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("模型综合对比报告\n")
        f.write("="*50 + "\n\n")
        
        f.write("参与对比的模型:\n")
        for model_name in model_names:
            if model_name in all_results:
                f.write(f"- {model_name}\n")
        f.write("\n")
        
        f.write("性能对比表:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'模型':<12} {'准确率':<10} {'精确率':<10} {'召回率':<10} {'F1分数':<10} {'误报率':<10}\n")
        f.write("-" * 80 + "\n")
        
        for model_name in model_names:
            if model_name in all_results:
                results = all_results[model_name]
                f.write(f"{model_name:<12} "
                       f"{results['accuracy']:<10.4f} "
                       f"{results['precision']:<10.4f} "
                       f"{results['recall']:<10.4f} "
                       f"{results['f1_score']:<10.4f} "
                       f"{results['false_positive_rate']:<10.4f}\n")
        
        f.write("-" * 80 + "\n\n")
        
        f.write("各指标最佳模型:\n")
        for metric, (model_name, value) in best_models.items():
            f.write(f"{metric.replace('_', ' ').title()}: {model_name} ({value:.4f})\n")
        f.write("\n")
        
        f.write("综合性能排名:\n")
        for i, (model_name, score) in enumerate(ranked_models, 1):
            f.write(f"{i}. {model_name}: {score:.4f}\n")
        
        f.write(f"\n报告生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"📄 对比报告已保存到: {filename}")