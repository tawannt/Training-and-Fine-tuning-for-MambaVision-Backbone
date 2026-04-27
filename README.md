# Training and Fine-tuning based on MambaVision Backbone
Training MambaVision-based model on Object Detection and Semantic Segmentation with subdataset. Also, fine-tuning + training MambaVision-based model for Face Parsing on LaPa Dataset.


## Fine-tuning: MambaVision with UperNet for Face Parsing (LaPa Dataset)

This repository also includes the fine-tuning of the **MambaVision** backbone integrated with the **UperNet** architecture for Face Parsing on the **LaPa Dataset**. 

### 🔗 Model Links
* **Hugging Face Model Page**: [tawannt/mamba-vision-b-upernet-for-face-parsing](https://huggingface.co/tawannt/mamba-vision-b-upernet-for-face-parsing)
* **Direct Download Link**: [mambavision_b_upernet_face_parsing.pth](https://huggingface.co/tawannt/mamba-vision-b-upernet-for-face-parsing/resolve/main/mambavision_b_upernet_face_parsing.pth)

### 📊 Performance Metrics
The model was evaluated on the test set of the LaPa dataset with the following global metrics:

| Metric | Score |
| --- | --- |
| **aAcc** | 97.44% |
| **mIoU** | 75.48% |
| **mAcc** | 85.47% |
| **mDice** | 85.40% |
| **mFscore** | 85.40% |
| **mPrecision** | 85.36% |
| **mRecall** | 85.47% |

**Inference Time:** ~0.4626s/img | **Data Time:** 0.0036s

### 🎯 Per-Class Metrics

| Class | IoU (%) | Acc (%) | Dice (%) | Fscore (%) | Precision (%) | Recall (%) |
| --- | --- | --- | --- | --- | --- | --- |
| **background** | 97.80 | 99.04 | 98.89 | 98.89 | 98.74 | 99.04 |
| **skin** | 91.77 | 95.16 | 95.71 | 95.71 | 96.26 | 95.16 |
| **left_eyebrow** | 69.87 | 82.01 | 82.26 | 82.26 | 82.51 | 82.01 |
| **right_eyebrow**| 68.77 | 82.62 | 81.49 | 81.49 | 80.40 | 82.62 |
| **left_eye** | 72.58 | 84.06 | 84.11 | 84.11 | 84.17 | 84.06 |
| **right_eye** | 71.86 | 83.59 | 83.63 | 83.63 | 83.67 | 83.59 |
| **nose** | 84.32 | 92.16 | 91.50 | 91.50 | 90.84 | 92.16 |
| **upper_lip** | 56.34 | 73.56 | 72.08 | 72.08 | 70.65 | 73.56 |
| **inner_mouth** | 64.14 | 75.80 | 78.15 | 78.15 | 80.66 | 75.80 |
| **lower_lip** | 61.63 | 76.91 | 76.26 | 76.26 | 75.62 | 76.91 |
| **hair** | 91.16 | 95.27 | 95.37 | 95.37 | 95.47 | 95.27 |


---

## Fine-tuning: MambaVision with Mask2Former for Face Parsing (LaPa Dataset)

This repository includes the fine-tuning of the **MambaVision** backbone integrated with the **Mask2Former** architecture for Face Parsing on the **LaPa Dataset**. 

### 🔗 Model Links
* **Hugging Face Model Page**: [tawannt/mamba-vision-b-mask2former-for-face-parsing](https://huggingface.co/tawannt/mamba-vision-b-mask2former-for-face-parsing)
* **Direct Download Link**: [mambavision_b_mask2former_face_parsing.pth](https://huggingface.co/tawannt/mamba-vision-b-mask2former-for-face-parsing/resolve/main/mambavision_b_mask2former_face_parsing.pth)

### 📊 Performance Metrics
The model was evaluated on the test set of the LaPa dataset with the following global metrics:

| Metric | Score |
| --- | --- |
| **aAcc** | 96.46% |
| **mIoU** | 81.60% |
| **mAcc** | 90.60% |
| **mDice** | 89.70% |
| **mFscore** | 89.70% |
| **mPrecision** | 88.82% |
| **mRecall** | 90.60% |

**Inference Time:** ~0.2738s/img | **Data Time:** 0.0035s

### 🎯 Per-Class Metrics

| Class | IoU (%) | Acc (%) | Dice (%) | Fscore (%) | Precision (%) | Recall (%) |
| --- | --- | --- | --- | --- | --- | --- |
| **background** | 96.16 | 97.82 | 98.04 | 98.04 | 98.26 | 97.82 |
| **skin** | 91.32 | 95.69 | 95.46 | 95.46 | 95.24 | 95.69 |
| **left_eyebrow** | 76.21 | 87.89 | 86.50 | 86.50 | 85.15 | 87.89 |
| **right_eyebrow**| 75.46 | 88.59 | 86.01 | 86.01 | 83.58 | 88.59 |
| **left_eye** | 78.07 | 89.30 | 87.69 | 87.69 | 86.13 | 89.30 |
| **right_eye** | 77.24 | 89.09 | 87.16 | 87.16 | 85.31 | 89.09 |
| **nose** | 90.29 | 95.88 | 94.90 | 94.90 | 93.93 | 95.88 |
| **upper_lip** | 74.39 | 85.83 | 85.31 | 85.31 | 84.81 | 85.83 |
| **inner_mouth** | 76.79 | 86.88 | 86.87 | 86.87 | 86.87 | 86.88 |
| **lower_lip** | 76.60 | 87.26 | 86.75 | 86.75 | 86.25 | 87.26 |
| **hair** | 85.13 | 92.42 | 91.97 | 91.97 | 91.52 | 92.42 |

### ⚙️ Training Configuration (Optimal Settings)

The optimal configuration used for training the model based on `mmsegmentation`:
- **Environment:** Google Colab (GPU: T4, 16GB RAM)
- **Optimizer Strategy (Phase 3 Micro-LR):**
  - Scheduler: `PolyLR` (power=0.9)
  - Backbone Learning Rate Multiplier: `0.01`, `0.1`, `0.2`
  - Query Embed / Query Feat / Level Embed LR Multiplier: `1.0`
- **Data & Batch:**
  - Dataset: `BaseSegDataset` targeting LaPa format
  - Batch Size: `2` (Accumulative counts: `8` -> Virtual Batch Size: `16`)

### Sample Inference
![Sample Inference](./img/MambaVision%20-%20Finetuning/Mask2Former/mask2former.png)