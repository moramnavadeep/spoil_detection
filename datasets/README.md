# 🥗 Food Freshness Dataset

This repository includes a curated sample subset (`datasets/sample_dataset/`) containing representative test images across all fresh and spoiled agricultural produce categories.

---

## 📊 Dataset Structure

```
datasets/
├── sample_dataset/                  # Curated benchmark dataset (105 images, 7.4MB)
│   ├── Fresh/
│   │   ├── FreshApple/
│   │   ├── FreshBanana/
│   │   ├── FreshBellpepper/
│   │   ├── FreshBittergroud/
│   │   ├── FreshCapsicum/
│   │   ├── FreshCarrot/
│   │   ├── FreshCucumber/
│   │   ├── FreshMango/
│   │   ├── FreshOkra/
│   │   ├── FreshOrange/
│   │   ├── FreshPotato/
│   │   ├── FreshStrawberry/
│   │   └── FreshTomato/
│   └── Rotten/
│       ├── RottenApple/
│       ├── RottenBanana/
│       ├── RottenBellpepper/
│       ├── RottenBittergroud/
│       ├── RottenCapsicum/
│       ├── RottenCarrot/
│       ├── RottenCucumber/
│       └── RottenMango/
└── README.md
```

---

## 📥 Download Full 4.1GB Dataset

To download the full Kaggle dataset (**60,700+ high-resolution images**), run the automated download script:

```bash
python scripts/download_dataset.py
```

Or download directly via Python:
```python
import kagglehub

# Download latest dataset version
path = kagglehub.dataset_download("ulnnproject/food-freshness-dataset")
print("Path to dataset:", path)
```

**Kaggle Source**: [ulnnproject/food-freshness-dataset](https://www.kaggle.com/datasets/ulnnproject/food-freshness-dataset)

---

## 🏋️ Training with Full Dataset

Once downloaded, train or fine-tune the CNN:
```bash
python train.py --dataset-path "<PATH_TO_DATASET>/versions/1/Dataset" --epochs 15 --batch-size 32
```
