"""
DeepSpaceVision - Data Augmentation (Məlumat Artırma)
=====================================================
Astronomik şəkilləri augmentasiya edərək datasetin ölçüsünü artırır.
Kosmik şəkillərə xas augmentasiya texnikaları tətbiq edir.
"""

import os
import random
from pathlib import Path

import cv2
import numpy as np


def apply_brightness_contrast(image, brightness=0, contrast=0):
    """
    Parlaqlıq və kontrast tənzimləmə.
    Kosmik şəkillər fərqli parlaqlıq səviyyələrində ola bilər.

    Args:
        image: Giriş şəkili (numpy array)
        brightness: Parlaqlıq dəyişikliyi (-100 ilə +100)
        contrast: Kontrast dəyişikliyi (-100 ilə +100)

    Returns:
        Tənzimlənmiş şəkil
    """
    # Kontrast: 131*(contrast + 127)/(127*(131-contrast))
    if contrast != 0:
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)
        image = cv2.addWeighted(image, alpha_c, image, 0, gamma_c)

    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow
        image = cv2.addWeighted(image, alpha_b, image, 0, gamma_b)

    return np.clip(image, 0, 255).astype(np.uint8)


def add_star_noise(image, n_stars=50, max_brightness=255):
    """
    Şəkilə təsadüfi "ulduz" nöqtələri əlavə edir.
    Bu, real astronomik şəkillərdəki arxa plan ulduzlarını simulyasiya edir.

    Args:
        image: Giriş şəkili
        n_stars: Əlavə ediləcək ulduz sayı
        max_brightness: Maksimum parlaqlıq
    """
    result = image.copy()
    h, w = result.shape[:2]

    for _ in range(n_stars):
        x = random.randint(0, w - 1)
        y = random.randint(0, h - 1)
        brightness = random.randint(150, max_brightness)
        size = random.randint(1, 3)

        cv2.circle(result, (x, y), size, (brightness, brightness, brightness), -1)

    return result


def simulate_light_pollution(image, intensity=0.15):
    """
    İşıq çirklənməsi simulyasiyası.
    Şəkilin bir tərəfinə yüngül parıltı əlavə edir.

    Args:
        image: Giriş şəkili
        intensity: Parıltı intensivliyi (0-1)
    """
    result = image.copy().astype(np.float32)
    h, w = result.shape[:2]

    # Gradient yaradır (bir tərəfdən parıltı)
    gradient = np.zeros((h, w), dtype=np.float32)
    direction = random.choice(['left', 'right', 'top', 'bottom'])

    if direction == 'left':
        gradient = np.tile(np.linspace(1, 0, w), (h, 1))
    elif direction == 'right':
        gradient = np.tile(np.linspace(0, 1, w), (h, 1))
    elif direction == 'top':
        gradient = np.tile(np.linspace(1, 0, h), (w, 1)).T
    else:
        gradient = np.tile(np.linspace(0, 1, h), (w, 1)).T

    # Rəng tonu əlavə et (narıncı/sarı — tipik işıq çirklənməsi)
    light = np.zeros_like(result)
    light[:, :, 0] = gradient * 40 * intensity   # B
    light[:, :, 1] = gradient * 60 * intensity   # G
    light[:, :, 2] = gradient * 80 * intensity   # R

    result = result + light * 255
    return np.clip(result, 0, 255).astype(np.uint8)


def random_rotate(image, labels=None, max_angle=180):
    """
    Şəkili təsadüfi bucaqda döndərir.
    Kosmik şəkillərdə istiqamət vacib deyil, ona görə
    geniş bucaq döndərmə istifadə oluna bilər.

    Args:
        image: Giriş şəkili
        labels: YOLO formatında etiketlər (opsional)
        max_angle: Maksimum dönmə bucağı

    Returns:
        (döndərülmüş_şəkil, yenilənmiş_etiketlər)
    """
    h, w = image.shape[:2]
    angle = random.uniform(-max_angle, max_angle)

    # Dönmə matrisi
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h),
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=(0, 0, 0))

    return rotated, labels  # Etiketlər üçün əlavə dönüşüm lazımdır


def augment_dataset(images_dir, labels_dir, output_images_dir, output_labels_dir,
                    augmentations_per_image=3):
    """
    Bütün dataseti augmentasiya edir.
    Hər şəkil üçün müəyyən sayda augmentasiya edilmiş variant yaradır.

    Tətbiq olunan augmentasiyalar:
    1. Parlaqlıq/kontrast dəyişikliyi
    2. Ulduz səs-küyü əlavəsi
    3. İşıq çirklənməsi simulyasiyası
    4. Təsadüfi dönmə
    5. Üfüqi/şaquli əksetdirmə

    Args:
        images_dir: Orijinal şəkillər qovluğu
        labels_dir: Orijinal etiketlər qovluğu
        output_images_dir: Augmentasiya edilmiş şəkillər çıxışı
        output_labels_dir: Augmentasiya edilmiş etiketlər çıxışı
        augmentations_per_image: Hər şəkil üçün neçə variant
    """
    Path(output_images_dir).mkdir(parents=True, exist_ok=True)
    Path(output_labels_dir).mkdir(parents=True, exist_ok=True)

    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    images = [f for f in Path(images_dir).iterdir()
              if f.suffix.lower() in image_extensions]

    print(f"🔄 Augmentasiya başlayır...")
    print(f"   Orijinal şəkillər: {len(images)}")
    print(f"   Hər biri üçün: {augmentations_per_image} variant")
    print(f"   Ümumi çıxış: {len(images) * augmentations_per_image} yeni şəkil")

    total_created = 0

    for img_path in images:
        image = cv2.imread(str(img_path))
        if image is None:
            continue

        # Uyğun etiketi tap
        label_path = Path(labels_dir) / (img_path.stem + '.txt')
        label_content = ""
        if label_path.exists():
            with open(label_path, 'r') as f:
                label_content = f.read()

        for aug_idx in range(augmentations_per_image):
            augmented = image.copy()

            # Təsadüfi augmentasiya seçimləri
            if random.random() > 0.5:
                brightness = random.randint(-30, 30)
                contrast = random.randint(-20, 20)
                augmented = apply_brightness_contrast(augmented, brightness, contrast)

            if random.random() > 0.6:
                augmented = add_star_noise(augmented, n_stars=random.randint(10, 80))

            if random.random() > 0.7:
                augmented = simulate_light_pollution(augmented,
                                                      intensity=random.uniform(0.05, 0.2))

            if random.random() > 0.5:
                augmented, _ = random_rotate(augmented, max_angle=180)

            if random.random() > 0.5:
                augmented = cv2.flip(augmented, 1)  # Üfüqi

            if random.random() > 0.5:
                augmented = cv2.flip(augmented, 0)  # Şaquli

            # Saxla
            aug_name = f"{img_path.stem}_aug{aug_idx}{img_path.suffix}"
            cv2.imwrite(str(Path(output_images_dir) / aug_name), augmented)

            # Etiketi kopyala
            if label_content:
                aug_label_name = f"{img_path.stem}_aug{aug_idx}.txt"
                with open(Path(output_labels_dir) / aug_label_name, 'w') as f:
                    f.write(label_content)

            total_created += 1

    print(f"\n✅ Augmentasiya tamamlandı!")
    print(f"   Yaradılan: {total_created} yeni şəkil")
    print(f"   Çıxış: {output_images_dir}")
