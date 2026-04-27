# Fine-tuning: MambaVision with UperNet for Face Parsing (LaPa Dataset)

This repository also includes the fine-tuning of the **MambaVision** backbone integrated with the **UperNet** architecture for Face Parsing on the **LaPa Dataset**. 

## 🔗 Model Links
* **Hugging Face Model Page**: [tawannt/mamba-vision-b-upernet-for-face-parsing](https://huggingface.co/tawannt/mamba-vision-b-upernet-for-face-parsing)
* **Direct Download Link**: [mambavision_b_upernet_face_parsing.pth](https://huggingface.co/tawannt/mamba-vision-b-upernet-for-face-parsing/resolve/main/mambavision_b_upernet_face_parsing.pth)

## 📊 Performance Metrics
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

## 🎯 Per-Class Metrics

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
