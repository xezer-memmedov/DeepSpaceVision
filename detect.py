"""
DeepSpaceVision - Aşkarlama / İnferens Skripti
================================================
Öyrədilmiş YOLOv8 modelini istifadə edərək şəkil və videolarda
dərin kosmik obyektləri aşkarlayır.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import torch
from ultralytics import YOLO


# Sinif rəngləri (BGR)
CLASS_COLORS = {
    'nebula': (255, 100, 50),       # Mavi-bənövşəyi
    'galaxy': (50, 255, 150),       # Yaşıl
    'star_cluster': (50, 150, 255), # Narıncı-sarı
}

# Sinif emoji-ləri
CLASS_EMOJIS = {
    'nebula': '🌌',
    'galaxy': '🌀',
    'star_cluster': '⭐',
}


def load_model(weights_path, device=None):
    """Modeli yükləyir."""
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"📦 Model yüklənir: {weights_path}")
    print(f"🖥️  Cihaz: {device}")

    model = YOLO(weights_path)
    return model


def draw_detections(image, results, conf_threshold=0.25):
    """
    Aşkarlanan obyektləri şəkil üzərində çəkir.
    Bounding box, sinif adı, etibar dərəcəsi göstərilir.
    """
    annotated = image.copy()
    detections = []

    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue

        for box in boxes:
            # Koordinatları al
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = result.names[cls_id]

            if conf < conf_threshold:
                continue

            # Rəng seç
            color = CLASS_COLORS.get(cls_name, (255, 255, 255))
            emoji = CLASS_EMOJIS.get(cls_name, '🔵')

            # Bounding box çək
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Etiket yaz
            label = f"{cls_name} {conf:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            label_y = max(y1, label_size[1] + 10)

            # Etiket arxa fonu
            cv2.rectangle(annotated,
                          (x1, label_y - label_size[1] - 10),
                          (x1 + label_size[0] + 10, label_y + 5),
                          color, -1)

            # Etiket mətni
            cv2.putText(annotated, label,
                        (x1 + 5, label_y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2)

            detections.append({
                'class': cls_name,
                'confidence': conf,
                'bbox': [x1, y1, x2, y2],
                'emoji': emoji
            })

    return annotated, detections


def detect_image(model, source, conf_threshold=0.25, save_dir='results'):
    """
    Tək şəkildə obyekt aşkarlama.
    """
    print(f"\n🔍 Şəkil analiz edilir: {source}")

    # Şəkili oxu
    image = cv2.imread(source)
    if image is None:
        print(f"❌ Şəkil oxunmadı: {source}")
        return None

    h, w = image.shape[:2]
    print(f"📐 Ölçü: {w}x{h}")

    # Aşkarlama
    start_time = time.time()
    results = model(image, conf=conf_threshold, verbose=False)
    inference_time = time.time() - start_time

    # Nəticələri çək
    annotated, detections = draw_detections(image, results, conf_threshold)

    # Statistika göstər
    print(f"⏱️  İnferens vaxtı: {inference_time:.3f}s")
    print(f"🎯 Aşkarlanan obyektlər: {len(detections)}")

    for det in detections:
        print(f"   {det['emoji']} {det['class']}: {det['confidence']:.2%}")

    # Nəticəni saxla
    os.makedirs(save_dir, exist_ok=True)
    filename = Path(source).stem + '_detected' + Path(source).suffix
    save_path = os.path.join(save_dir, filename)
    cv2.imwrite(save_path, annotated)
    print(f"💾 Nəticə saxlanıldı: {save_path}")

    return annotated, detections


def detect_video(model, source, conf_threshold=0.25, save_dir='results'):
    """
    Videoda obyekt aşkarlama (kadr-kadr).
    """
    print(f"\n🎥 Video analiz edilir: {source}")

    # Videonu aç
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"❌ Video açılmadı: {source}")
        return None

    # Video xüsusiyyətləri
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"📐 Ölçü: {width}x{height}")
    print(f"🎞️  FPS: {fps}, Ümumi kadr: {total_frames}")

    # Çıxış videosunu yarat
    os.makedirs(save_dir, exist_ok=True)
    filename = Path(source).stem + '_detected.mp4'
    save_path = os.path.join(save_dir, filename)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

    frame_count = 0
    total_detections = 0
    detection_summary = {}

    print(f"\n🔄 İşlənir...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Aşkarlama
        results = model(frame, conf=conf_threshold, verbose=False)
        annotated, detections = draw_detections(frame, results, conf_threshold)

        # Kadr nömrəsi və FPS göstər
        info_text = f"Frame: {frame_count}/{total_frames} | Objects: {len(detections)}"
        cv2.putText(annotated, info_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Statistikalar
        total_detections += len(detections)
        for det in detections:
            cls = det['class']
            detection_summary[cls] = detection_summary.get(cls, 0) + 1

        # Çıxışa yaz
        out.write(annotated)

        # Hər 50 kadra bir status göstər
        if frame_count % 50 == 0:
            progress = frame_count / total_frames * 100
            print(f"   [{progress:5.1f}%] Kadr {frame_count}/{total_frames} - "
                  f"Bu kadrdakı obyektlər: {len(detections)}")

    cap.release()
    out.release()

    # Yekun statistika
    print(f"\n{'=' * 50}")
    print(f"✅ Video analizi tamamlandı!")
    print(f"{'=' * 50}")
    print(f"📊 Ümumi kadr: {frame_count}")
    print(f"🎯 Ümumi aşkarlama: {total_detections}")
    print(f"\n📈 Sinif statistikası:")
    for cls, count in sorted(detection_summary.items(), key=lambda x: x[1], reverse=True):
        emoji = CLASS_EMOJIS.get(cls, '🔵')
        print(f"   {emoji} {cls}: {count} dəfə")
    print(f"\n💾 Nəticə saxlanıldı: {save_path}")

    return save_path, detection_summary


def detect_batch(model, source_dir, conf_threshold=0.25, save_dir='results'):
    """
    Qovluqdakı bütün şəkilləri analiz edir.
    """
    print(f"\n📁 Qovluq analiz edilir: {source_dir}")

    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.fits'}
    images = [f for f in Path(source_dir).iterdir()
              if f.suffix.lower() in image_extensions]

    print(f"🖼️  Tapılan şəkillər: {len(images)}")

    all_detections = []
    for i, img_path in enumerate(images, 1):
        print(f"\n[{i}/{len(images)}] {img_path.name}")
        result = detect_image(model, str(img_path), conf_threshold, save_dir)
        if result:
            _, detections = result
            all_detections.extend(detections)

    # Ümumi nəticə
    print(f"\n{'=' * 50}")
    print(f"✅ Toplu analiz tamamlandı!")
    print(f"   Şəkillər: {len(images)}")
    print(f"   Ümumi obyektlər: {len(all_detections)}")

    return all_detections


def main():
    parser = argparse.ArgumentParser(
        description='🔭 DeepSpaceVision - Dərin Kosmik Obyektlərin Aşkarlanması',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
İstifadə Nümunələri:
  # Tək şəkil analizi
  python detect.py --source data/images/nebula.jpg --weights models/best.pt

  # Video analizi
  python detect.py --source video.mp4 --weights models/best.pt

  # Qovluqdakı bütün şəkillər
  python detect.py --source data/images/ --weights models/best.pt

  # Aşağı etibar həddi ilə (daha çox aşkarlama)
  python detect.py --source image.jpg --weights models/best.pt --conf 0.15
        """
    )

    parser.add_argument('--source', type=str, required=True,
                        help='Şəkil, video və ya qovluq yolu')
    parser.add_argument('--weights', type=str, default='models/best.pt',
                        help='Model çəkiləri (default: models/best.pt)')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Etibar həddi (default: 0.25)')
    parser.add_argument('--save-dir', type=str, default='results',
                        help='Nəticə qovluğu (default: results)')
    parser.add_argument('--device', type=str, default=None,
                        help='Cihaz: cuda, cpu (default: auto)')

    args = parser.parse_args()
    source = args.source

    # Modeli yüklə
    model = load_model(args.weights, args.device)

    # Mənbə tipini müəyyən et
    if os.path.isdir(source):
        detect_batch(model, source, args.conf, args.save_dir)
    elif source.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.wmv')):
        detect_video(model, source, args.conf, args.save_dir)
    else:
        detect_image(model, source, args.conf, args.save_dir)


if __name__ == '__main__':
    main()
