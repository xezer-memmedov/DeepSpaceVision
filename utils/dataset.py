"""
📘 DeepSpaceVision — dataset.py (TAM İZAHLI)
=============================================

Bu fayl nə edir?
-----------------
Kosmik şəkilləri (nebula, galaxy, star cluster) YOLOv8 modeli
üçün hazırlayır. 4 əsas iş görür:

  1️⃣ download_file()        → Şəkilləri internetdən yükləyir
  2️⃣ split_dataset()        → Şəkilləri train/val/test-ə bölür
  3️⃣ create_sample_labels() → Test üçün nümunə etiketlər yaradır
  4️⃣ verify_dataset()       → Datasetin düzgün olduğunu yoxlayır

YOLO modeli bu qovluq strukturunu gözləyir:
---------------------------------------------
  data/
  ├── images/
  │   ├── train/   ← Modelin öyrəndiyi şəkillər (70%)
  │   ├── val/     ← Öyrətmə zamanı yoxlama (20%)
  │   └── test/    ← Son qiymətləndirmə (10%)
  └── labels/
      ├── train/   ← Hər şəklə uyğun .txt etiket faylı
      ├── val/
      └── test/

YOLO Etiket Formatı (hər sətir = 1 obyekt):
---------------------------------------------
  <sinif_id> <x_merkez> <y_merkez> <en> <boy>

  Bütün dəyərlər 0 ilə 1 arasındadır (normallaşdırılmış).

  Nümunə: "0 0.5 0.5 0.3 0.4"
    0   → sinif 0 = nebula (dumanlıq)
    0.5 → x mərkəz = şəklin 50%-i (ortada)
    0.5 → y mərkəz = şəklin 50%-i (ortada)
    0.3 → en = şəklin 30%-i qədər geniş
    0.4 → boy = şəklin 40%-i qədər hündür

  Siniflər:
    0 = nebula       (dumanlıq)
    1 = galaxy       (qalaktika)
    2 = star_cluster (ulduz topası)
"""


# ====================================================
# 📦 LAZIMI KİTABXANALAR (import-lar)
# ====================================================
# Hər kitabxana nə üçün lazımdır:

import os
# → Fayl/qovluq əməliyyatları
# → Nümunə: os.path.exists("fayl.txt") — fayl varmı?
# → Nümunə: os.makedirs("qovluq") — qovluq yarat

import random
# → Təsadüfi rəqəmlər və qarışdırma
# → Nümunə: random.shuffle(list) — siyahını qarışdır
# → Nümunə: random.randint(0, 2) — 0-2 arası rəqəm

import shutil
# → Faylları kopyalama/daşıma
# → Nümunə: shutil.copy2("a.jpg", "b/a.jpg") — kopyala

from pathlib import Path
# → Fayl yolları ilə rahat işləmək
# → Nümunə: Path("images/a.jpg").stem → "a" (adsız)
# → Nümunə: Path("images/a.jpg").suffix → ".jpg" (uzantı)
# → Nümunə: Path("images/").iterdir() → qovluqdakı fayllar

import requests
# → İnternetdən fayl yükləmək (HTTP sorğuları)
# → Nümunə: requests.get("https://...") → faylı al

import yaml
# → YAML formatında konfiqurasiya fayllarını oxumaq
# → Nümunə: yaml.safe_load(file) → dict-ə çevir

from tqdm import tqdm
# → Yükləmə zamanı progress bar göstərir
# → Nümunə: for item in tqdm(items): ...
# → Ekranda: [████████░░░░] 65% 130MB/200MB


# ====================================================
# 1️⃣ FAYL YÜKLƏMƏ FUNKSİYASI
# ====================================================
# NƏ EDİR:  İnternetdən faylı yükləyib kompüterə saxlayır
# NİYƏ:    Dataset şəkillərini və ya model fayllarını almaq üçün
# NÜMUNƏ:  download_file("https://data.com/space.zip", "data/space.zip")

def download_file(url, dest_path, chunk_size=8192):
    """
    Faylı URL-dən yükləyir (progress bar ilə).

    Parametrlər:
    ------------
    url : str
        Yükləmə linki.
        Nümunə: "https://github.com/.../dataset.zip"

    dest_path : str
        Faylın saxlanacağı yer.
        Nümunə: "data/dataset.zip"

    chunk_size : int (default: 8192)
        Hər dəfə neçə bayt (byte) oxunacaq.
        8192 = 8 KiloBayt

        Niyə hissə-hissə? Çünki 1GB fayl varsa,
        hamısını birdən RAM-a yükləmək yaddaşı doldurar.
        Chunk ilə hissə-hissə yükləyib yazırıq.
    """

    # ADDIM 1: Serverə sorğu göndər
    # stream=True → faylı birdəfəyə yox, hissə-hissə al
    response = requests.get(url, stream=True)

    # ADDIM 2: Faylın ümumi ölçüsünü öyrən
    # Server cavab başlığında (header) faylın ölçüsünü bildirir
    # Bu, progress bar-ın % göstərməsi üçün lazımdır
    total_size = int(response.headers.get('content-length', 0))
    # Əgər server ölçünü bilməsə, 0 olacaq

    # ADDIM 3: Faylı aç və yaz
    # 'wb' = write + binary (ikili rejimdə yaz)
    # Şəkillər, videolar, zip fayllar ikili formatdadır
    with open(dest_path, 'wb') as f:

        # ADDIM 4: Progress bar yarat
        # total = ümumi ölçü, unit='B' = bayt, unit_scale=True = KB/MB göstər
        with tqdm(total=total_size, unit='B', unit_scale=True,
                  desc=Path(dest_path).name) as pbar:

            # ADDIM 5: Hissə-hissə oxu
            # iter_content → serverdən chunk_size bayt oxuyur
            for chunk in response.iter_content(chunk_size=chunk_size):
                f.write(chunk)           # Hissəni fayla yaz
                pbar.update(len(chunk))  # Progress bar-ı güncəllə
                # Ekranda belə görünür:
                # dataset.zip: 65%|██████░░░░| 130MB/200MB [00:15<00:08]

    print(f"✅ Yükləndi: {dest_path}")


# ====================================================
# 2️⃣ DATASETİ BÖLMƏ FUNKSİYASI
# ====================================================
# NƏ EDİR:  Şəkilləri train (70%), val (20%), test (10%) bölür
# NİYƏ:    Model eyni datadan həm öyrənib həm test olunmamalıdır
#           (əzbərləmənin — overfitting-in qarşısını alır)
# NÜMUNƏ:  split_dataset("raw/images", "raw/labels", "data/")
#
# REAL HƏYAT ANALOGIYASI:
# → Tələbə (model) dərsliklə öyrənir (train)
# → Ev tapşırığı ilə yoxlayır (val)
# → Final imtahanla qiymətləndirilir (test)
# → Dərslik = train, ev tapşırığı = val, imtahan = test

def split_dataset(images_dir, labels_dir, output_dir,
                  train_ratio=0.7, val_ratio=0.2, test_ratio=0.1,
                  seed=42):
    """
    Dataseti train/val/test bölmələrinə ayırır.

    Parametrlər:
    ------------
    images_dir : str
        Orijinal şəkillərin olduğu qovluq.
        Nümunə: "raw_data/images"

    labels_dir : str
        YOLO etiketlərinin olduğu qovluq.
        Nümunə: "raw_data/labels"

    output_dir : str
        Çıxış qovluğu — nəticə buraya yazılacaq.
        Nümunə: "data"

    train_ratio : float (default: 0.7)
        Train nisbəti. 0.7 = 70% şəkil train-ə gedəcək.

    val_ratio : float (default: 0.2)
        Validation nisbəti. 0.2 = 20%.

    test_ratio : float (default: 0.1)
        Test nisbəti. 0.1 = 10%.

    seed : int (default: 42)
        Təsadüfi toxum — təkrarlanabilirlik üçün.
        Eyni seed ilə işlətsən, hər dəfə eyni bölgü olacaq.
        42 rəqəmi konvensional olaraq istifadə olunur.
    """

    # ADDIM 1: Nisbətlərin cəmini yoxla
    # 0.7 + 0.2 + 0.1 = 1.0 olmalıdır, yoxsa xəta ver
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, \
        "Nisbətlərin cəmi 1.0 olmalıdır!"
    # 1e-5 = 0.00001 (float hesablamasında kiçik xətaları nəzərə alır)

    # ADDIM 2: Təsadüfi toxumu qur
    # Bu, nəticələri təkrarlana bilən edir
    random.seed(seed)

    # ADDIM 3: Bütün şəkil fayllarını tap
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    # set {} istifadə edirik — axtarış list []-dən daha sürətli

    images = [
        f                                      # Faylı siyahıya əlavə et
        for f in Path(images_dir).iterdir()    # Qovluqdakı hər faylı gəz
        if f.suffix.lower() in image_extensions  # Yalnız şəkil faylları
    ]
    # .suffix = ".jpg", .lower() = böyük/kiçik hərfə görə (.JPG → .jpg)

    # ADDIM 4: Siyahını təsadüfi qarışdır
    # Niyə? Əgər fayllar adlarına görə sıralanıbsa:
    #   galaxy_001.jpg, galaxy_002.jpg, ..., nebula_001.jpg, ...
    # O zaman qarışdırmasaq, bütün galaxy-lər train-ə,
    # bütün nebula-lar test-ə düşə bilər → ədalətsiz!
    random.shuffle(images)

    # ADDIM 5: Bölmə ölçülərini hesabla
    n = len(images)                    # Ümumi şəkil sayı
    n_train = int(n * train_ratio)     # 100 şəkildən 70-i train
    n_val = int(n * val_ratio)         # 100 şəkildən 20-si val
    # Qalanı test                       # 100 - 70 - 20 = 10 test

    # ADDIM 6: Şəkilləri 3 hissəyə ayır
    splits = {
        'train': images[:n_train],                    # Əvvəldən 70-ə qədər
        'val':   images[n_train:n_train + n_val],     # 70-dən 90-a qədər
        'test':  images[n_train + n_val:],            # 90-dan sona qədər
    }

    print(f"\n📊 Dataset Bölməsi:")
    print(f"   Ümumi: {n} şəkil")

    # ADDIM 7: Hər bölmə üçün faylları kopyala
    for split_name, split_images in splits.items():
        # Çıxış qovluqlarını yarat
        img_dir = Path(output_dir) / 'images' / split_name
        # Nümunə: data/images/train/

        lbl_dir = Path(output_dir) / 'labels' / split_name
        # Nümunə: data/labels/train/

        img_dir.mkdir(parents=True, exist_ok=True)
        # parents=True → data/ yoxdursa, onu da yarat
        # exist_ok=True → qovluq artıq varsa, xəta vermə

        lbl_dir.mkdir(parents=True, exist_ok=True)

        count = 0  # Etiketli şəkil sayğacı

        for img_path in split_images:
            # Şəkili kopyala
            shutil.copy2(img_path, img_dir / img_path.name)
            # copy2 = fayl + metadata (yaradılma tarixi və s.) kopyalayır
            # img_path.name = "nebula_001.jpg" (tam ad)

            # Uyğun etiket faylını tap
            # nebula_001.jpg → nebula_001.txt
            label_name = img_path.stem + '.txt'
            # .stem = uzantısız ad: "nebula_001"

            label_path = Path(labels_dir) / label_name

            if label_path.exists():
                # Etiket mövcuddursa, kopyala
                shutil.copy2(label_path, lbl_dir / label_name)
                count += 1
            # Əgər etiket yoxdursa, YOLO bu şəkili "arxa plan"
            # (heç bir obyekt yoxdur) kimi qəbul edəcək

        print(f"   {split_name:5s}: {len(split_images):4d} şəkil, "
              f"{count:4d} etiketli")
        # :5s → 5 simvol genişliyində string
        # :4d → 4 rəqəm genişliyində integer

    print(f"\n✅ Dataset bölməsi tamamlandı: {output_dir}")


# ====================================================
# 3️⃣ NÜMUNƏ ETİKET YARATMA
# ====================================================
# NƏ EDİR:  Test/demo üçün təsadüfi etiket faylları yaradır
# NİYƏ:    Real dataset olmadan kodu test etmək üçün
#
# ⚠️ DİQQƏT: Bu REAL layihədə istifadə olunmur!
# Real layihədə annotasiya alətləri ilə əl ilə etiketlənir:
#   - Roboflow (ən asan — web-based)
#   - LabelImg (pulsuz — desktop)
#   - CVAT (Intel-in pulsuz aləti)

def create_sample_labels(images_dir, labels_dir, class_names):
    """
    Demo üçün təsadüfi YOLO etiketləri yaradır.

    Parametrlər:
    ------------
    images_dir : str
        Şəkillərin olduğu qovluq.

    labels_dir : str
        Etiketlərin yazılacağı qovluq.

    class_names : list
        Sinif adları.
        Nümunə: ['nebula', 'galaxy', 'star_cluster']
    """

    # Etiketlər qovluğunu yarat (yoxdursa)
    Path(labels_dir).mkdir(parents=True, exist_ok=True)

    # Şəkil fayllarını tap
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    images = [f for f in Path(images_dir).iterdir()
              if f.suffix.lower() in image_extensions]

    for img_path in images:
        # Etiket faylının yolunu müəyyən et
        # nebula_001.jpg → nebula_001.txt
        label_path = Path(labels_dir) / (img_path.stem + '.txt')

        # Hər şəkil üçün 1-3 arası təsadüfi obyekt yarat
        n_objects = random.randint(1, 3)

        with open(label_path, 'w') as f:
            for _ in range(n_objects):
                # _ = dəyişən adı lazım deyil, sadəcə n dəfə təkrarla

                # Təsadüfi sinif (0=nebula, 1=galaxy, 2=star_cluster)
                cls_id = random.randint(0, len(class_names) - 1)

                # Təsadüfi bounding box koordinatları
                # 0.1-0.9 arası → kənarlardan bir az uzaqda
                x_center = random.uniform(0.1, 0.9)
                y_center = random.uniform(0.1, 0.9)

                # Ölçü: şəklin 5%-40%-i arası
                width = random.uniform(0.05, 0.4)
                height = random.uniform(0.05, 0.4)

                # Fayla yaz: "sinif x y en boy"
                # :.6f = 6 onluq rəqəm (0.523481)
                f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} "
                        f"{width:.6f} {height:.6f}\n")

    print(f"📝 {len(images)} nümunə etiket yaradıldı: {labels_dir}")


# ====================================================
# 4️⃣ DATASET YOXLAMA FUNKSİYASI
# ====================================================
# NƏ EDİR:  Datasetin düzgün strukturda olduğunu yoxlayır
# NİYƏ:    Öyrətməyə başlamazdan əvvəl problem tapır
#           Əksik etiketlər → model pis nəticə verir!
# NÜMUNƏ:  verify_dataset("data/")
#           Çıxış: "✅ Dataset düzgün strukturdadır!"

def verify_dataset(data_dir):
    """
    Datasetin düzgünlüyünü yoxlayır.

    Yoxlayır:
      ✓ Qovluqlar mövcuddur (data/images/train, val, test)
      ✓ Hər şəklin etiketi var
      ✓ Statistika göstərir

    Parametrlər:
    ------------
    data_dir : str
        Data qovluğunun yolu. Nümunə: "data"

    Qaytarır:
    ---------
    bool : True = problem yoxdur, False = problem var
    """

    print(f"\n🔍 Dataset yoxlanılır: {data_dir}")
    issues = []    # Tapılan problemlər siyahısı
    stats = {}     # Hər bölmənin statistikası

    # Hər bölməni yoxla
    for split in ['train', 'val', 'test']:

        # Qovluq yollarını qur
        img_dir = Path(data_dir) / 'images' / split
        lbl_dir = Path(data_dir) / 'labels' / split

        # Qovluq mövcuddurmu?
        if not img_dir.exists():
            issues.append(f"❌ Qovluq tapılmadı: {img_dir}")
            continue  # Bu bölməni atla, növbətiyə keç

        # Faylları say
        images = list(img_dir.glob('*'))
        # glob('*') = qovluqdakı bütün fayllar

        labels = list(lbl_dir.glob('*.txt')) if lbl_dir.exists() else []
        # glob('*.txt') = yalnız .txt faylları
        # if lbl_dir.exists() = qovluq varsa axtar, yoxsa boş siyahı

        # Etiket olmayan şəkilləri tap
        image_stems = {f.stem for f in images}
        # set comprehension — hər faylın adını (uzantısız) çıxar
        # {'nebula_001', 'galaxy_002', 'galaxy_003'}

        label_stems = {f.stem for f in labels}
        # {'nebula_001', 'galaxy_002'}

        # Fərq = etiketsiz şəkillər
        missing_labels = image_stems - label_stems
        # {'galaxy_003'} — bu şəklin etiketi yoxdur!

        # Statistika saxla
        stats[split] = {
            'images': len(images),
            'labels': len(labels),
            'missing_labels': len(missing_labels)
        }

        if missing_labels:
            issues.append(
                f"⚠️  {split}: {len(missing_labels)} şəklin etiketi yoxdur"
            )

    # Gözəl cədvəl formatında göstər
    print(f"\n📊 Dataset Statistikası:")
    print(f"   {'Bölmə':8s} {'Şəkil':>8s} {'Etiket':>8s} {'Əksik':>8s}")
    print(f"   {'-'*36}")
    for split, s in stats.items():
        print(f"   {split:8s} {s['images']:>8d} {s['labels']:>8d} "
              f"{s['missing_labels']:>8d}")

    # Nəticə
    if issues:
        print(f"\n⚠️  Problemlər:")
        for issue in issues:
            print(f"   {issue}")
    else:
        print(f"\n✅ Dataset düzgün strukturdadır!")

    return len(issues) == 0
