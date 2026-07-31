"""
DeepSpaceVision - Model Öyrətmə Skripti
========================================
YOLOv8 modelini dərin kosmik obyektlərin aşkarlanması üçün öyrədir.
Destəklənən obyektlər: Dumanlıq (Nebula), Qalaktika (Galaxy), Ulduz Topası (Star Cluster)
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

import yaml
import torch
from ultralytics import YOLO


def get_device():
    """Mövcud cihazı (GPU/CPU) müəyyən edir."""
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"🖥️  GPU tapıldı: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        device = "cpu"
        print("⚠️  GPU tapılmadı, CPU istifadə olunacaq (yavaş olacaq)")
    return device


def load_config(config_path):
    """YAML konfiqurasiya faylını yükləyir."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_data_yaml(config, project_root):
    """
    Ultralytics YOLO üçün data.yaml faylı yaradır.
    Bu fayl modelin dataseti haradan tapacağını bildirir.
    """
    data_yaml = {
        'path': str(Path(project_root) / 'data'),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': config['nc'],
        'names': config['names']
    }

    yaml_path = Path(project_root) / 'data' / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    print(f"📄 Data YAML yaradıldı: {yaml_path}")
    return str(yaml_path)


def train(args):
    """Əsas öyrətmə funksiyası."""

    print("=" * 60)
    print("🚀 DeepSpaceVision - Model Öyrətmə")
    print("=" * 60)

    # Cihazı müəyyən et
    device = get_device()

    # Konfiqurasiyanı yüklə
    config = load_config(args.data)
    print(f"\n📋 Konfiqurasiya: {args.data}")
    print(f"   Sinif sayı: {config['nc']}")
    print(f"   Siniflər: {config['names']}")

    # Project root
    project_root = Path(__file__).parent

    # Data YAML yarad
    data_yaml_path = create_data_yaml(config, project_root)

    # Modeli yüklə
    if args.weights and os.path.exists(args.weights):
        print(f"\n📦 Mövcud çəkilər yüklənir: {args.weights}")
        model = YOLO(args.weights)
    else:
        model_size = args.model_size  # n, s, m, l, x
        print(f"\n📦 YOLOv8{model_size} modeli yüklənir (pretrained)...")
        model = YOLO(f'yolov8{model_size}.pt')

    # Öyrətmə parametrləri
    train_params = {
        'data': data_yaml_path,
        'epochs': args.epochs,
        'batch': args.batch,
        'imgsz': args.imgsz,
        'device': device,
        'workers': args.workers,
        'patience': args.patience,
        'save': True,
        'save_period': 10,
        'project': str(project_root / 'runs'),
        'name': f'deep_space_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        'exist_ok': False,
        'pretrained': True,
        'optimizer': 'auto',
        'verbose': True,
        'seed': 42,
        'cos_lr': True,
        'plots': True,
    }

    # Augmentasiya parametrləri
    if 'augmentation' in config:
        aug = config['augmentation']
        train_params.update({
            'hsv_h': aug.get('hsv_h', 0.015),
            'hsv_s': aug.get('hsv_s', 0.7),
            'hsv_v': aug.get('hsv_v', 0.4),
            'degrees': aug.get('degrees', 15.0),
            'translate': aug.get('translate', 0.1),
            'scale': aug.get('scale', 0.5),
            'fliplr': aug.get('fliplr', 0.5),
            'flipud': aug.get('flipud', 0.5),
            'mosaic': aug.get('mosaic', 1.0),
            'mixup': aug.get('mixup', 0.1),
        })

    print(f"\n🏋️ Öyrətmə başlayır...")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch: {args.batch}")
    print(f"   Image Size: {args.imgsz}")
    print(f"   Device: {device}")
    print("-" * 60)

    # Öyrətməni başlat
    results = model.train(**train_params)

    # Nəticələri göstər
    print("\n" + "=" * 60)
    print("✅ Öyrətmə tamamlandı!")
    print("=" * 60)

    # Ən yaxşı modelin yolunu göstər
    best_model_path = Path(train_params['project']) / train_params['name'] / 'weights' / 'best.pt'
    print(f"\n📁 Ən yaxşı model: {best_model_path}")
    print(f"📊 Nəticələr: {Path(train_params['project']) / train_params['name']}")

    # Modeli models/ qovluğuna kopyala
    models_dir = project_root / 'models'
    models_dir.mkdir(exist_ok=True)
    if best_model_path.exists():
        import shutil
        dest = models_dir / 'best.pt'
        shutil.copy2(best_model_path, dest)
        print(f"📋 Model kopyalandı: {dest}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='🚀 DeepSpaceVision - Dərin Kosmik Obyektlərin Aşkarlanması üçün YOLOv8 Öyrətmə',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
İstifadə Nümunələri:
  python train.py --data configs/deep_space.yaml --epochs 100
  python train.py --data configs/deep_space.yaml --epochs 50 --batch 8 --model-size s
  python train.py --weights models/best.pt --data configs/deep_space.yaml --epochs 50
        """
    )

    parser.add_argument('--data', type=str, default='configs/deep_space.yaml',
                        help='Dataset konfiqurasiya faylı (default: configs/deep_space.yaml)')
    parser.add_argument('--weights', type=str, default=None,
                        help='Öncədən öyrədilmiş çəkilər (fine-tuning üçün)')
    parser.add_argument('--model-size', type=str, default='n', choices=['n', 's', 'm', 'l', 'x'],
                        help='YOLOv8 model ölçüsü: n(ano), s(mall), m(edium), l(arge), x(large)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Öyrətmə epoch sayı (default: 100)')
    parser.add_argument('--batch', type=int, default=16,
                        help='Batch ölçüsü (default: 16)')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Şəkil ölçüsü (default: 640)')
    parser.add_argument('--workers', type=int, default=4,
                        help='Data loader işçi sayı (default: 4)')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience (default: 20)')

    args = parser.parse_args()

    # Öyrətməni başlat
    train(args)


if __name__ == '__main__':
    main()
