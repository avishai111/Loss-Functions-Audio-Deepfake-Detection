# Audio Deepfake Detection Loss Functions

## 📖 Overview
This repository provides implementations of various loss functions designed for **Audio Deepfake Detection**, focusing on robust feature learning and classification. The methods include:

- **AOCloss (Adaptive Centroid Shift Loss)**: A one-class learning framework that adapts a centroid for bonafide samples while distancing spoof samples.
- **OCSoftmax (One-Class Softmax)**: A loss function that classifies bonafide and spoofed audio based on one-class learning for bonafide samples.
- **AMSoftmax (Additive Margin Softmax Loss)**: A margin-based loss function to enhance class separability.

Each loss function is implemented in PyTorch and can be seamlessly integrated into deep learning models for audio deepfake detection.

---

## 🛠️ Loss Functions

### 1️⃣ **AOCloss (Adaptive Centroid Shift Loss)**

#### 📌 Description
AOCloss continuously adapts a centroid to represent bonafide audio embeddings while maximizing the distance of spoof embeddings.

#### 🏗️ **Initialization**
```python
from AOC_loss import AOCloss

# Initialize with embedding dimension
criterion = AOCloss(embedding_dim=512)
```

#### 🔄 **Usage**
```python
loss = criterion(embeddings, labels)
```
- `embeddings`: Tensor of shape `(batch_size, embedding_dim)`
- `labels`: Binary tensor where `1` represents bonafide samples and `0` represents spoof samples.

#### 🔧 **Centroid Update**
```python
criterion.update_centroid(bonafide_embeddings)
```
The centroid is automatically updated during the forward pass.

---

### 2️⃣ **OCSoftmax (One-Class Softmax Loss)**

#### 📌 Description
OCSoftmax is designed for **one-class classification**, based on one-class learning for bonafide samples.

#### 🏗️ **Initialization**
```python
from OCSoftmax_loss import OCSoftmax

criterion = OCSoftmax(feat_dim=512, r_real=0.9, r_fake=0.5, alpha=20.0)
```

#### 🔄 **Usage**
```python
loss = criterion(embeddings, labels)
```
- `embeddings`: Feature matrix `(batch_size, feat_dim)`
- `labels`: Ground truth labels (0: Spoof, 1: Bonafide)

#### 🔍 **Inference**
```python
scores = criterion.inference(embeddings)
```
Returns similarity scores for classification.

---

### 3️⃣ **AMSoftmax (Additive Margin Softmax Loss)**

#### 📌 Description
AMSoftmax enforces **angular margin constraints** in softmax loss, improving classification performance.

#### 🏗️ **Initialization (Version 1 - Embedding Based)**
```python
from AMSoftmax_loss2 import AMSoftmaxLoss

criterion = AMSoftmaxLoss(embedding_dim=512, no_classes=2, scale=30.0, margin=0.4)
```

#### 🔄 **Usage**
```python
loss, margin_logits, logits = criterion(embeddings, labels)
```
- `embeddings`: Tensor `(batch_size, embedding_dim)`
- `labels`: Class labels (0: Spoof, 1: Bonafide)

#### 🏗️ **Initialization (Version 2 - Linear Layer Based)**
```python
from AMSoftmax_loss import AMSoftmaxLoss
criterion = AMSoftmaxLoss(in_features=512, out_features=2, s=30.0, m=0.4)
```

#### 🔄 **Usage**
```python
loss = criterion(embeddings, labels)
```
---

## 📄 Citation
This repository is based on the following research papers:
```
@inproceedings{kim24b_interspeech,
  title     = {One-class learning with adaptive centroid shift for audio deepfake detection},
  author    = {Hyun Myung Kim and Kangwook Jang and Hoirin Kim},
  year      = {2024},
  booktitle = {Interspeech 2024},
  pages     = {4853--4857},
  doi       = {10.21437/Interspeech.2024-177},
  issn      = {2958-1796},
}
```

```
@article{zhang2021one,
  title={One-class learning towards synthetic voice spoofing detection},
  author={Zhang, You and Jiang, Fei and Duan, Zhiyao},
  journal={IEEE Signal Processing Letters},
  volume={28},
  pages={937--941},
  year={2021},
  publisher={IEEE}
}
```

```
@article{wang2018additive,
  title={Additive margin softmax for face verification},
  author={Wang, Feng and Cheng, Jian and Liu, Weiyang and Liu, Haijun},
  journal={IEEE Signal Processing Letters},
  volume={25},
  number={7},
  pages={926--930},
  year={2018},
  publisher={IEEE}
}
```

---

## ❓ Troubleshooting
- **`ValueError: Centroid has not been initialized`**:
  - Ensure the batch contains bonafide samples.
- **Negative Loss**:
  - This is expected due to cosine similarity range `-1` to `1`.

---

## 🙋‍♂️ Support
For issues or questions, feel free to open an issue in this repository.

