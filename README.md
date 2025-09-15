# Machine Learning Projects Portfolio

This repository contains applied machine learning projects built with **PyTorch** and **scikit-learn**.  
The projects demonstrate end-to-end workflows: data preparation, model training, evaluation, fairness checks, and production-ready exports.

---

## Projects

### Multimodal Engagement Prediction (PyTorch)
Predicts engagement likelihood from text synopses and tabular metadata such as genre, maturity rating, and duration.  
- BiGRU with additive attention for text  
- MLP encoder for tabular features  
- Fusion head with dropout and layer normalization  
- Metrics: AUC, Accuracy, Brier Score, ECE (calibration), per-genre AUC  
- Exports: TorchScript and ONNX  

Folder: `multimodal/`  
Script: `multimodal_ctr.py`

---

### Vision Transformer on CIFAR-10 (PyTorch)
Implements a Vision Transformer (ViT) from scratch for image classification.  
- Patch embeddings, class token, positional encoding  
- Multi-head self-attention encoder blocks  
- Data augmentation with AutoAugment and RandomErasing  
- Metrics: Top-1 Accuracy, Confusion Matrix  
- Exports: TorchScript  
- Includes attention rollout visualization  

Folder: `vision_transformer/`  
Script: `vit_cifar10.py`

---

### Talent Acquisition Analytics (scikit-learn)
Applies machine learning to optimize the hiring funnel using synthetic ATS/HRIS-style data.  
- Candidate success prediction with calibrated classification  
- Fairness audit by group with threshold tuning  
- Time-to-hire regression with gradient boosting  
- Sourcing channel effectiveness and lift analysis  
- Funnel anomaly detection with Isolation Forest  
- Exports: joblib models and model card  

Folder: `talent_analytics/`  
Script: `talent_analytics_sklearn.py`

---

## Example Results
- Multimodal Engagement: AUC ~0.87, well-calibrated (ECE ~0.03)  
- Vision Transformer: CIFAR-10 Top-1 Accuracy ~80%  
- Talent Analytics: Candidate Success AUC ~0.85, Time-to-Hire MAE ~5 days  

---

## Technology Stack
- Python 3.10+  
- PyTorch for deep learning  
- scikit-learn for classical machine learning  
- TorchScript and ONNX for model export  
- joblib for persistence  
- Calibration metrics and fairness reporting  

---

## Repository Structure
