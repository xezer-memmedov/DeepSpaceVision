"""
DeepSpaceVision - Model Qiymətləndirmə Skripti
=================================================
Öyrədilmiş modeli test dataseti üzərində qiymətləndirir.
mAP, Precision, Recall, F1 metriklərini hesablayır.
"""

import argparse
import os
import json
from pathlib import Path
from datetime import datetime

import torch
from ultralytics import YOLO


def evaluate(args):
    """Model qiymətləndirmə."""

    print("=" * 60)
    print("📊 DeepSpaceVision - Model Qiymətləndirmə")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  Cihaz: {device}")

    # Modeli yüklə
    print(f"📦 Model yüklənir: {args.weights}")
    model = YOLO(args.weights)

    # Qiymətləndirmə
    print(f"\n🔍 Qiymətləndirmə başlayır...")
    print(f"   Data: {args.data}")
    print(f"   Image Size: {args.imgsz}")
    print("-" * 60)

    results = model.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        split=args.split,
        verbose=True,
        plots=True,
    )

    # Nəticələri göstər
    print("\n" + "=" * 60)
    print("📊 QİYMƏTLƏNDİRMƏ NƏTİCƏLƏRİ")
    print("=" * 60)

    metrics = {
        'mAP@0.5': float(results.box.map50),
        'mAP@0.5:0.95': float(results.box.map),
        'Precision': float(results.box.mp),
        'Recall': float(results.box.mr),
    }

    for metric, value in metrics.items():
        bar = '█' * int(value * 30) + '░' * (30 - int(value * 30))
        print(f"   {metric:20s}: {bar} {value:.4f}")

    # Per-class nəticələr
    print(f"\n📈 Sinif üzrə nəticələr:")
    print(f"   {'Sinif':15s} {'mAP@0.5':>10s} {'mAP@0.5:0.95':>15s}")
    print(f"   {'-'*42}")

    if hasattr(results.box, 'maps') and results.box.maps is not None:
        class_names = model.names
        for i, class_map in enumerate(results.box.maps):
            name = class_names.get(i, f'class_{i}')
            print(f"   {name:15s} {results.box.map50:>10.4f} {class_map:>15.4f}")

    # Nəticələri fayla yaz
    os.makedirs('results', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"results/evaluation_{timestamp}.json"

    report = {
        'timestamp': timestamp,
        'model': args.weights,
        'data': args.data,
        'device': device,
        'metrics': metrics,
    }

    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n💾 Hesabat saxlanıldı: {report_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='📊 DeepSpaceVision - Model Qiymətləndirmə',
        epilog="""
İstifadə:
  python evaluate.py --weights models/best.pt --data configs/deep_space.yaml
  python evaluate.py --weights models/best.pt --data configs/deep_space.yaml --split test
        """
    )

    parser.add_argument('--weights', type=str, default='models/best.pt',
                        help='Model çəkiləri')
    parser.add_argument('--data', type=str, default='configs/deep_space.yaml',
                        help='Dataset konfiqurasiyası')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Şəkil ölçüsü')
    parser.add_argument('--batch', type=int, default=16,
                        help='Batch ölçüsü')
    parser.add_argument('--split', type=str, default='val',
                        choices=['val', 'test'],
                        help='Qiymətləndirmə bölməsi')

    args = parser.parse_args()
    evaluate(args)


if __name__ == '__main__':
    main()
