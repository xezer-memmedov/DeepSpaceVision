"""
DeepSpaceVision - Vizuallaşdırma Alətləri
==========================================
Aşkarlama nəticələrini, öyrətmə metriklərini və
konfuziya matrisini vizuallaşdırır.
"""

import os
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec


# Sinif rəngləri (matplotlib formatı: 0-1 arası RGB)
CLASS_COLORS_MPL = {
    'nebula': (0.2, 0.4, 1.0),       # Mavi
    'galaxy': (0.2, 1.0, 0.6),       # Yaşıl
    'star_cluster': (1.0, 0.6, 0.2), # Narıncı
}

CLASS_EMOJIS = {
    'nebula': '🌌',
    'galaxy': '🌀',
    'star_cluster': '⭐',
}


def plot_detection_results(image_path, detections, save_path=None, figsize=(14, 10)):
    """
    Aşkarlama nəticələrini matplotlib ilə vizuallaşdırır.
    Bounding box-lar, sinif adları və etibar dərəcələri göstərilir.

    Args:
        image_path: Orijinal şəkil yolu
        detections: Aşkarlama nəticələri siyahısı
                   [{'class': 'nebula', 'confidence': 0.95, 'bbox': [x1,y1,x2,y2]}, ...]
        save_path: Nəticə faylının saxlanma yolu (None = göstər)
        figsize: Fiqur ölçüsü
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(image)
    ax.set_title(f'🔭 DeepSpaceVision Aşkarlama — {len(detections)} obyekt tapıldı',
                 fontsize=16, fontweight='bold', pad=15)

    for det in detections:
        cls = det['class']
        conf = det['confidence']
        x1, y1, x2, y2 = det['bbox']
        color = CLASS_COLORS_MPL.get(cls, (1, 1, 1))
        emoji = CLASS_EMOJIS.get(cls, '')

        # Bounding box
        width = x2 - x1
        height = y2 - y1
        rect = patches.Rectangle(
            (x1, y1), width, height,
            linewidth=2, edgecolor=color,
            facecolor=(*color, 0.15)  # Yarı-şəffaf dolgu
        )
        ax.add_patch(rect)

        # Etiket
        label = f"{emoji} {cls} ({conf:.0%})"
        ax.text(x1, y1 - 8, label,
                fontsize=11, fontweight='bold',
                color='white',
                bbox=dict(boxstyle='round,pad=0.3',
                          facecolor=color, alpha=0.85))

    ax.axis('off')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor='black', edgecolor='none')
        print(f"💾 Vizuallaşdırma saxlanıldı: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_training_metrics(results_csv_path, save_path=None):
    """
    Öyrətmə metriklərini (loss, mAP, precision, recall) çəkir.
    Ultralytics YOLO-nun results.csv faylını oxuyur.

    Bu qrafik öyrətmənin necə getdiyini göstərir:
    - Loss azalırsa → model öyrənir ✅
    - mAP artırsa → model daha dəqiq olur ✅
    - Loss artırsa → overfitting ola bilər ⚠️

    Args:
        results_csv_path: Ultralytics results.csv faylının yolu
        save_path: Saxlanma yolu
    """
    import pandas as pd

    df = pd.read_csv(results_csv_path)
    # Sütun adlarındakı boşluqları sil
    df.columns = df.columns.str.strip()

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle('🚀 DeepSpaceVision — Öyrətmə Nəticələri',
                 fontsize=18, fontweight='bold', y=0.98)

    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    # --- 1. Box Loss ---
    ax1 = fig.add_subplot(gs[0, 0])
    if 'train/box_loss' in df.columns:
        ax1.plot(df['epoch'], df['train/box_loss'], 'b-', label='Train', linewidth=2)
    if 'val/box_loss' in df.columns:
        ax1.plot(df['epoch'], df['val/box_loss'], 'r--', label='Val', linewidth=2)
    ax1.set_title('📦 Box Loss', fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- 2. Class Loss ---
    ax2 = fig.add_subplot(gs[0, 1])
    if 'train/cls_loss' in df.columns:
        ax2.plot(df['epoch'], df['train/cls_loss'], 'b-', label='Train', linewidth=2)
    if 'val/cls_loss' in df.columns:
        ax2.plot(df['epoch'], df['val/cls_loss'], 'r--', label='Val', linewidth=2)
    ax2.set_title('🏷️ Class Loss', fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # --- 3. DFL Loss ---
    ax3 = fig.add_subplot(gs[0, 2])
    if 'train/dfl_loss' in df.columns:
        ax3.plot(df['epoch'], df['train/dfl_loss'], 'b-', label='Train', linewidth=2)
    if 'val/dfl_loss' in df.columns:
        ax3.plot(df['epoch'], df['val/dfl_loss'], 'r--', label='Val', linewidth=2)
    ax3.set_title('📐 DFL Loss', fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # --- 4. mAP@0.5 ---
    ax4 = fig.add_subplot(gs[1, 0])
    if 'metrics/mAP50(B)' in df.columns:
        ax4.plot(df['epoch'], df['metrics/mAP50(B)'], 'g-', linewidth=2)
        ax4.fill_between(df['epoch'], 0, df['metrics/mAP50(B)'], alpha=0.1, color='green')
    ax4.set_title('🎯 mAP@0.5', fontweight='bold')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('mAP')
    ax4.set_ylim(0, 1)
    ax4.grid(True, alpha=0.3)

    # --- 5. Precision & Recall ---
    ax5 = fig.add_subplot(gs[1, 1])
    if 'metrics/precision(B)' in df.columns:
        ax5.plot(df['epoch'], df['metrics/precision(B)'], 'b-',
                 label='Precision', linewidth=2)
    if 'metrics/recall(B)' in df.columns:
        ax5.plot(df['epoch'], df['metrics/recall(B)'], 'orange',
                 label='Recall', linewidth=2)
    ax5.set_title('📏 Precision & Recall', fontweight='bold')
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Dəyər')
    ax5.set_ylim(0, 1)
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # --- 6. mAP@0.5:0.95 ---
    ax6 = fig.add_subplot(gs[1, 2])
    if 'metrics/mAP50-95(B)' in df.columns:
        ax6.plot(df['epoch'], df['metrics/mAP50-95(B)'], 'm-', linewidth=2)
        ax6.fill_between(df['epoch'], 0, df['metrics/mAP50-95(B)'],
                         alpha=0.1, color='magenta')
    ax6.set_title('🎯 mAP@0.5:0.95', fontweight='bold')
    ax6.set_xlabel('Epoch')
    ax6.set_ylabel('mAP')
    ax6.set_ylim(0, 1)
    ax6.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"💾 Qrafik saxlanıldı: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_class_distribution(labels_dir, class_names, save_path=None):
    """
    Datasetdəki sinif paylanmasını göstərir.
    Hər sinfin neçə dəfə göründüyünü bar chart ilə vizuallaşdırır.

    Args:
        labels_dir: YOLO etiketlər qovluğu
        class_names: Sinif adları {0: 'nebula', 1: 'galaxy', ...}
        save_path: Saxlanma yolu
    """
    class_counts = {name: 0 for name in class_names.values()}

    for label_file in Path(labels_dir).glob('*.txt'):
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    cls_id = int(parts[0])
                    cls_name = class_names.get(cls_id, f'class_{cls_id}')
                    class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

    # Qrafik
    fig, ax = plt.subplots(figsize=(10, 6))

    names = list(class_counts.keys())
    counts = list(class_counts.values())
    colors = [CLASS_COLORS_MPL.get(n, (0.5, 0.5, 0.5)) for n in names]

    bars = ax.bar(names, counts, color=colors, edgecolor='white', linewidth=1.5)

    # Dəyərləri bar-ların üstünə yaz
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(count), ha='center', va='bottom', fontweight='bold', fontsize=14)

    ax.set_title('📊 Sinif Paylanması', fontsize=16, fontweight='bold')
    ax.set_xlabel('Sinif', fontsize=12)
    ax.set_ylabel('Sayı', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"💾 Qrafik saxlanıldı: {save_path}")
    else:
        plt.show()

    plt.close()


def create_detection_grid(images_with_detections, cols=3, save_path=None):
    """
    Bir neçə aşkarlama nəticəsini bir şəbəkədə göstərir.
    Bu, toplu analizin nəticələrini tez görmək üçün faydalıdır.

    Args:
        images_with_detections: [(image_path, detections), ...] siyahısı
        cols: Sütun sayı
        save_path: Saxlanma yolu
    """
    n = len(images_with_detections)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 6 * rows))
    fig.suptitle('🔭 DeepSpaceVision — Toplu Aşkarlama Nəticələri',
                 fontsize=18, fontweight='bold')

    if rows == 1:
        axes = [axes] if cols == 1 else axes
    axes_flat = np.array(axes).flatten()

    for idx, ax in enumerate(axes_flat):
        if idx < n:
            img_path, detections = images_with_detections[idx]
            image = cv2.imread(img_path)
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                ax.imshow(image)
                ax.set_title(f"{Path(img_path).name}\n"
                             f"({len(detections)} obyekt)",
                             fontsize=10)
        ax.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"💾 Şəbəkə saxlanıldı: {save_path}")
    else:
        plt.show()

    plt.close()
