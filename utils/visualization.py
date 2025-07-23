"""
可视化工具
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from config import get_plot_path

def plot_confusion_matrix(cm, label_map, true_labels, pred_labels, model_name):
    """绘制混淆矩阵"""
    unique_labels = sorted(np.unique(np.concatenate([true_labels, pred_labels])))
    
    plt.figure(figsize=(12, 10))
    
    if len(unique_labels) <= 15:
        # 小规模数据集：显示标签名称
        tick_labels = [label_map.get(i, f"Class_{i}") for i in unique_labels]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                   xticklabels=tick_labels, yticklabels=tick_labels,
                   cbar_kws={'label': '样本数量'})
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
    else:
        # 大规模数据集：完全移除刻度线
        sns.heatmap(cm, annot=False, cmap="Blues", 
                   cbar_kws={'label': '样本数量'})
        # 移除坐标轴刻度线
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])
        # 添加中心化的轴标签
        plt.xlabel("Index of prediction categories", labelpad=15)
        plt.ylabel("Index of real categories", labelpad=15)
    
    plt.title(f"{model_name} confusion matrix ")
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(get_plot_path(f'confusion_matrix_{model_name.lower()}'))
    plt.show()

def show_prediction_examples(val_images, val_labels, predictions, label_map, model_name):
    """显示正确和错误识别的示例"""
    correct_idx = np.where(predictions == val_labels)[0][:5]
    incorrect_idx = np.where(predictions != val_labels)[0][:5]
    
    plt.figure(figsize=(15, 8))
    plt.suptitle(f"{model_name} Prediction Examples")
    
    for i, idx in enumerate(correct_idx, 1):
        plt.subplot(2, 5, i)
        if len(val_images[idx].shape) == 3:  # 彩色图像
            img_rgb = cv2.cvtColor(val_images[idx], cv2.COLOR_BGR2RGB)
            plt.imshow(img_rgb)
        else:  # 灰度图像
            plt.imshow(val_images[idx], cmap='gray')
        plt.title(f"True: {label_map[val_labels[idx]]}\nPred: {label_map[predictions[idx]]}", color='green')
        plt.axis('off')
    
    for i, idx in enumerate(incorrect_idx, 6):
        plt.subplot(2, 5, i)
        if len(val_images[idx].shape) == 3:  # 彩色图像
            img_rgb = cv2.cvtColor(val_images[idx], cv2.COLOR_BGR2RGB)
            plt.imshow(img_rgb)
        else:  # 灰度图像
            plt.imshow(val_images[idx], cmap='gray')
        plt.title(f"True: {label_map[val_labels[idx]]}\nPred: {label_map[predictions[idx]]}", color='red')
        plt.axis('off')
    
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(get_plot_path(f'prediction_examples_{model_name.lower()}'))
    plt.show()

def plot_performance_metrics(results, model_name):
    """绘制性能指标图表"""
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'false_positive_rate']
    values = [results[m] for m in metrics]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics, values)
    plt.title(f"{model_name} Performance Metrics")
    plt.ylim(0, 1)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.4f}', ha='center', va='bottom')
    plt.grid(axis='y', alpha=0.3)
    
    # 保存图片
    plt.savefig(get_plot_path(f'performance_metrics_{model_name.lower()}'))
    plt.show()

def plot_pca_analysis(model, model_name):
    """绘制PCA分析图"""
    if not hasattr(model, 'pca') or model.pca is None:
        return
        
    explained_variance_ratio = model.pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 个别组件解释方差
    ax1.bar(range(1, min(21, len(explained_variance_ratio) + 1)), explained_variance_ratio[:20])
    ax1.set_xlabel('Principal Component Number')
    ax1.set_ylabel('Proportion of variance explained')
    ax1.set_title('Proportion of variance explained by the top 20 principal components')
    ax1.grid(True, alpha=0.3)
    
    # 累积解释方差
    ax2.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, 'bo-')
    ax2.axhline(y=0.95, color='r', linestyle='--', label='95% variance line')
    ax2.set_xlabel('Number of principal components')
    ax2.set_ylabel('Cumulative proportion of variance explained')
    ax2.set_title('Cumulative proportion of variance explained')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(get_plot_path(f'pca_analysis_{model_name.lower()}'))
    plt.show()
    
    print(f"前10个主成分解释方差比例: {explained_variance_ratio[:10]}")
    print(f"使用{len(explained_variance_ratio)}个主成分，累积解释方差: {cumulative_variance_ratio[-1]:.4f}")

def plot_lda_analysis(model, model_name):
    """绘制LDA分析图"""
    if not hasattr(model, 'lda') or model.lda is None:
        return
        
    explained_variance_ratio = model.lda.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 个别组件解释方差
    ax1.bar(range(1, min(21, len(explained_variance_ratio) + 1)), explained_variance_ratio[:20])
    ax1.set_xlabel('判别成分编号')
    ax1.set_ylabel('解释方差比例')
    ax1.set_title('前20个判别成分解释方差比例')
    ax1.grid(True, alpha=0.3)
    
    # 累积解释方差
    ax2.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, 'bo-')
    ax2.axhline(y=0.95, color='r', linestyle='--', label='95%方差线')
    ax2.set_xlabel('判别成分数量')
    ax2.set_ylabel('累积解释方差比例')
    ax2.set_title('累积解释方差比例')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(get_plot_path(f'lda_analysis_{model_name.lower()}'))
    plt.show()
    
    print(f"前10个判别成分解释方差比例: {explained_variance_ratio[:10]}")
    print(f"使用{len(explained_variance_ratio)}个判别成分，累积解释方差: {cumulative_variance_ratio[-1]:.4f}")

def print_quantitative_comparison(results, model_name):
    """打印定量比较结果"""
    print(f"\n📊 Quantitative Comparison for {model_name}")
    print("-"*50)
    for key, value in results.items():
        if key == 'model_info':
            continue
        if 'time' in key:
            print(f"{key.replace('_', ' ').title()}: {value:.2f} seconds")
        else:
            print(f"{key.replace('_', ' ').title()}: {value:.4f} ({value*100:.2f}%)")