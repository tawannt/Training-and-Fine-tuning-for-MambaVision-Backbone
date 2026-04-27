# 🎭 Facial Landmark-based Data Preprocessing

> [!NOTE]
> **Purpose:** Raw facial images contain significant scale and spatial variations. To eliminate spatial variance and allow the model to focus on semantic features (eyes, nose, mouth), we applied a preprocessing pipeline based on **Facial Landmarks**. Inspired by the `RefineBB` technique, this ensures the face is always centered and maximizes the **Signal-to-Noise Ratio (SNR)**.

---

## ⚙️ Processing Pipeline

### Step 1: Initial Bounding Box Extraction
Using the 106 facial landmarks, denoted as $L = \{(x_i, y_i)\}_{i=1}^{106}$, the initial bounding box is determined by extracting the extreme coordinates:

* **Min Coordinates:**
  $$x_{min} = \min(x_i), \quad y_{min} = \min(y_i)$$
* **Max Coordinates:**
  $$x_{max} = \max(x_i), \quad y_{max} = \max(y_i)$$

The bounding box's width ($W$) and height ($H$) are simply derived from these points.

### Step 2: Bounding Box Refinement (1:1 Ratio)
To prevent geometric distortion during resizing, the model input must be a perfect square (1:1 ratio).

1. **Calculate Base Size:**
   $$S = \max(W, H)$$
2. **Apply Margin:**
   Keeping the face's center $(x_c, y_c)$ as the anchor, we apply a margin ratio of $m = 0.25$ (25%) to capture surrounding context (hair, neck) without cropping the face edges.
3. **Determine Optimal Size:**
   $$S_{new} = \lfloor S \times (1 + 0.25) \rfloor$$
   
The refined square bounding box $(x_1, y_1, x_2, y_2)$ is then built symmetrically around the center using $S_{new}$.

### Step 3: Zero-Padding and Resizing
If the refined bounding box exceeds the original image dimensions, **Constant Zero-Padding** is applied to fill the missing areas. This yields black pixels for RGB images and maps to the `Background` class for label masks, preserving the strict 1:1 ratio.

Finally, the images and masks are resized to the model's standard input size ($512 \times 512$) using two distinct interpolation methods:

* **RGB Images (Bilinear Interpolation):** Ensures smooth color transitions and preserves surface details.
* **Label Masks (Nearest-Neighbor Interpolation):** *Strictly applied* to preserve discrete categorical integer values ($0-10$ for the $11$ classes). This prevents the generation of meaningless decimal values that would corrupt Loss computations.

![Bounding Box Refinement](../../img/MambaVision%20-%20Finetuning/Data%20Engineering/visualize_bbox_refinement.png)
---

## 📉 Alignment Noise Reduction Analysis

To mathematically evaluate the effectiveness of the preprocessing pipeline, we compute the **Signal-to-Noise Ratio (SNR)** and **Spatial Variance**. The SNR for a single image $i$ is defined as the ratio of face pixels ($N_{face}$) to background pixels ($N_{bg}$):

$$SNR_i = \frac{N_{face}^{(i)}}{N_{bg}^{(i)}}$$

The Mean and Standard Deviation of SNR across the entire dataset $N$ are:

$$\mu_{SNR} = \frac{1}{N} \sum_{i=1}^{N} SNR_i \quad ; \quad \sigma_{SNR} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (SNR_i - \mu_{SNR})^2}$$

To measure spatial variance, we calculate the normalized center of mass for the face in each image:

$$cx_i = \frac{1}{W} \left( \frac{1}{N_{face}^{(i)}} \sum_{(x,y) \in Face} x \right), \quad cy_i = \frac{1}{H} \left( \frac{1}{N_{face}^{(i)}} \sum_{(x,y) \in Face} y \right)$$

And the global spatial variance of the face centers across the dataset:

$$Var_x = \frac{1}{N} \sum_{i=1}^{N} (cx_i - \mu_{cx})^2 \quad ; \quad Var_y = \frac{1}{N} \sum_{i=1}^{N} (cy_i - \mu_{cy})^2$$

**Results on the Training Set:**

| Metric | Raw Dataset | Aligned Dataset |
| :--- | :---: | :---: |
| **Mean SNR** | 0.4954 | **4.0267** |
| **Std SNR** | 0.3042 | **9.9422** |
| **X-Center Variance** | 0.00271 | **0.00106** |
| **Y-Center Variance** | 0.00657 | **0.00160** |

*The results show a massive $8\times$ increase in Mean SNR and a significant reduction in spatial variance, proving that the aligned dataset is much cleaner and strictly centered.*


---

## 🔗 Model Links
* **Hugging Face Model Page**: [tawannt/mambavision-b-mask2former-data-processing-face-parsing](https://huggingface.co/tawannt/mambavision-b-mask2former-data-processing-face-parsing)
* **Direct Download Link**: [mambavision_b_mask2former_data_processing_face_parsing.pth](https://huggingface.co/tawannt/mambavision-b-mask2former-data-processing-face-parsing/resolve/main/mambavision_b_mask2former_data_processing_face_parsing.pth)



## 🚀 Performance on Processed Test Set


### Convergence Plot
![Convergence Plot](../../img/MambaVision%20-%20Finetuning/Data%20Engineering/convergence_plot.png)

By applying this landmark-based preprocessing pipeline to align and crop the test set, the model's performance improves significantly. Below are the metrics evaluated on the **processed LaPa test set**:

### Global Metrics

| Metric | Score |
| --- | --- |
| **aAcc** | 96.38% |
| **mIoU** | 85.65% |
| **mAcc** | 91.67% |
| **mDice** | 92.16% |
| **mFscore** | 92.16% |
| **mPrecision** | 92.67% |
| **mRecall** | 91.67% |

**Inference Time:** ~0.2574s/img | **Data Time:** 0.0030s

### Per-Class Metrics

| Class | IoU (%) | Acc (%) | Dice (%) | Fscore (%) | Precision (%) | Recall (%) |
| --- | --- | --- | --- | --- | --- | --- |
| **background** | 93.70 | 96.39 | 96.75 | 96.75 | 97.11 | 96.39 |
| **skin** | 95.22 | 97.68 | 97.55 | 97.55 | 97.43 | 97.68 |
| **left_eyebrow** | 82.21 | 89.86 | 90.24 | 90.24 | 90.62 | 89.86 |
| **right_eyebrow**| 81.91 | 89.44 | 90.06 | 90.06 | 90.68 | 89.44 |
| **left_eye** | 84.01 | 89.47 | 91.31 | 91.31 | 93.23 | 89.47 |
| **right_eye** | 84.54 | 91.07 | 91.63 | 91.63 | 92.19 | 91.07 |
| **nose** | 93.81 | 96.64 | 96.80 | 96.80 | 96.97 | 96.64 |
| **upper_lip** | 76.94 | 86.21 | 86.97 | 86.97 | 87.74 | 86.21 |
| **inner_mouth** | 81.27 | 89.50 | 89.67 | 89.67 | 89.84 | 89.50 |
| **lower_lip** | 80.24 | 87.73 | 89.04 | 89.04 | 90.39 | 87.73 |
| **hair** | 88.27 | 94.39 | 93.77 | 93.77 | 93.15 | 94.39 |

### Inference Result
![Inference Result](../../img/MambaVision%20-%20Finetuning/Data%20Engineering/inference_result.png)