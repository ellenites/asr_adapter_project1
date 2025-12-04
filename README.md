 ### ASR Fellowship Challenge – Adapter Fine-Tuning

## Overview

This repository contains the solution for the **ASR Fellowship Challenge**. The goal is to improve the Word Error Rate (WER) of a pre-trained Automatic Speech Recognition (ASR) model on the **Afrivoice_Kinyarwanda health dataset** using **adapter modules** while keeping the base model frozen.

Adapters allow domain-specific fine-tuning without catastrophic forgetting by inserting small, trainable modules into a frozen pre-trained model.

---

## Repository Structure

```
.
├── base_transcriptions.txt        # Transcriptions of test set using base model
├── finetuned_transcriptions.txt   # Transcriptions of test set using adapter-finetuned model
├── adapters/                       # Directory containing trained adapter weights
│   └── adapter_weights.pt
├── base_model/                     # Base model weights
│   └── base_model_weights
├── src/                            # Source code for data loading, training, and evaluation
│   ├── data_preprocessing.py
│   ├── train_adapter.py
│   ├── evaluate.py
│   └── utils.py
├── report.pdf                      # Challenge report with results and methodology
└── README.md
```

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/ellenites/asr_adapter_project1.git
cd asr_adapter_project1
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

**Key dependencies include:**

* Python 3.8+
* PyTorch
* Hugging Face Transformers & Datasets

## Usage

### 1. Base Model Evaluation

Generate transcriptions for the test set using the pre-trained base model:

```bash
python src/evaluate.py --mode base --dataset_dir data/ --model_dir base_model/
```

### 2. Adapter Fine-Tuning

Train the adapter on the training dataset:

```bash
python src/train_adapter.py --dataset_dir data/ --adapter_dir adapters/
```

### 3. Fine-Tuned Evaluation

Generate transcriptions using the adapter-finetuned model:

```bash
python src/evaluate.py --mode finetuned --dataset_dir data/ --adapter_dir adapters/
```

---

## Results

* **Base Model WER:** 
* **Adapter Fine-Tuned Model WER:** 
* **Number of Trainable Parameters:** 

---

## Adapter Architecture

* Small trainable modules inserted into the frozen pre-trained ASR model.
* Efficient for domain-specific learning without forgetting general ASR knowledge.


## Reproducibility

1. Install dependencies and download the dataset.
2. Run `train_adapter.py` to train adapters.
3. Use `evaluate.py` to generate base and fine-tuned transcriptions.
4. Compare WERs using the provided scripts or any WER calculation tool.

---

## Contact

**Name:** Elleni Sisay
**Email:** elleni.sisay@aait.edu.et


