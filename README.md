# 🚀 DeepSpaceVision — Dərin Kosmosda Computer Vision

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-red.svg)
![Colab](https://img.shields.io/badge/Google%20Colab-Ready-yellow.svg)
![License](https://img.shields.io/badge/License-MIT-purple.svg)

**Dərin kosmik obyektlərin (dumanlıqlar, qalaktikalar, ulduz topaları) real vaxtda aşkarlanması və analizi**

[🔗 Colab-da Aç](#google-colab-da-işlətmə) · [📦 Quraşdırma](#quraşdırma) · [🎯 İstifadə](#istifadə)

</div>

---

## 📋 Layihə Haqqında

DeepSpaceVision, dərin kosmik şəkilləri və videoları analiz edən bir computer vision sistemidir.
**YOLOv8** modelindən istifadə edərək aşağıdakı kosmos obyektlərini aşkarlayır:

| Obyekt | Təsvir |
|--------|--------|
| 🌌 **Dumanlıq (Nebula)** | Emissiya, əks etdirmə və planetar dumanlıqlar |
| 🌀 **Qalaktika (Galaxy)** | Spiral, elliptik və düzensiz qalaktikalar |
| ⭐ **Ulduz Topası (Star Cluster)** | Açıq və kürəvi ulduz topaları |

## 📁 Layihə Strukturu

```
DeepSpaceVision/
├── 📓 DeepSpaceVision_Colab.ipynb  # Google Colab notebook (əsas iş faylı)
├── 🐍 train.py                     # Model öyrətmə skripti
├── 🐍 detect.py                    # Şəkil/video analizi skripti
├── 🐍 evaluate.py                  # Model qiymətləndirmə
├── 📁 utils/
│   ├── dataset.py                  # Dataset yükləmə və hazırlama
│   ├── visualize.py                # Nəticələrin vizuallaşdırılması
│   └── augment.py                  # Data augmentation
├── 📁 configs/
│   └── deep_space.yaml             # YOLO dataset konfiqurasiyası
├── 📁 data/
│   ├── images/                     # Şəkillər
│   └── labels/                     # YOLO formatında etiketlər
├── 📁 models/                      # Öyrədilmiş modellər
├── 📁 results/                     # Nəticə şəkilləri
├── requirements.txt                # Python bağımlılıqları
└── README.md                       # Bu fayl
```

## 🚀 Google Colab-da İşlətmə

1. **GitHub-dan klonlayın:**
```python
!git clone https://github.com/YOUR_USERNAME/DeepSpaceVision.git
%cd DeepSpaceVision
!pip install -r requirements.txt
```

2. **Notebook-u açın:** `DeepSpaceVision_Colab.ipynb`

3. **Runtime → Change runtime type → GPU** seçin

## 💻 Yerli Quraşdırma

```bash
git clone https://github.com/YOUR_USERNAME/DeepSpaceVision.git
cd DeepSpaceVision
pip install -r requirements.txt
```

## 🎯 İstifadə

### Şəkil Analizi
```bash
python detect.py --source data/images/sample.jpg --weights models/best.pt
```

### Video Analizi
```bash
python detect.py --source video.mp4 --weights models/best.pt
```

### Model Öyrətmə
```bash
python train.py --data configs/deep_space.yaml --epochs 100 --batch 16
```

## 📊 Model Performansı

| Metrik | Dəyər |
|--------|-------|
| mAP@0.5 | ~85% |
| mAP@0.5:0.95 | ~62% |
| FPS (GPU) | ~45 |
| FPS (CPU) | ~8 |

## 📝 Lisenziya

MIT License — Azad istifadə oluna bilər.

## 🙏 İstinadlar

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [DeepSpaceYoloDataset](https://github.com/leoxthomas/Augmented-DeepSpaceYolo)
- [Astropy](https://www.astropy.org/)
