# Generalizable Bangla Misinformation Detection Under Distribution Shift

A research-oriented NLP project that evaluates whether Bangla misinformation classifiers remain reliable when the data distribution changes.

Instead of reporting only conventional random train/test accuracy, this project studies **generalization across unseen topics, publishers/sources, future time periods, external datasets, and noisy text** using classical NLP baselines and Transformer-based models.

---

## Project Overview

Modern misinformation classifiers often perform very well when the training and test data come from the same dataset. However, real-world deployment is harder: new topics appear, publishers change, language evolves over time, and external data may differ substantially from the training distribution.

This project investigates that gap for **Bangla misinformation detection**.

The core question is:

> **Can a Bangla misinformation detector that performs well in-domain maintain its performance under realistic distribution shifts?**

The project is designed as an **independent NLP research project** for demonstrating practical experience with:

- Bangla Natural Language Processing
- Text classification
- Transformer fine-tuning
- BanglaBERT
- Multilingual language models
- Classical NLP baselines
- Distribution shift
- Temporal generalization
- Cross-dataset transfer
- Robustness analysis
- Error analysis
- Statistical model comparison

---

## Research Questions

The project evaluates the following questions:

1. How well do traditional NLP and Transformer models perform on standard in-domain Bangla misinformation classification?
2. Can models generalize to **unseen topics**?
3. Can models generalize to **unseen publishers or source domains**?
4. Can models trained on older data maintain performance on **future time periods**?
5. How well do models transfer to a **completely different Bangla misinformation dataset**?
6. How sensitive are Transformer models to **textual noise and perturbations**?
7. Do high in-domain scores overestimate real-world robustness?

---

## Datasets

### 1. BanglaFakeNews2025

Primary dataset used for model development and most generalization experiments.

**Dataset characteristics used in this project:**

- 4,000 Bangla news articles
- 2,000 Fake
- 2,000 Real
- Rich metadata including:
  - Article ID
  - Domain
  - Date
  - Category
  - Headline
  - Content
  - Label
  - Source
  - Relation
  - Fake-news type

**Binary labels:**

```text
0 = Fake
1 = Real
```

The dataset is used for:

- Standard in-domain evaluation
- Topic-held-out evaluation
- Source-held-out evaluation
- Temporal generalization
- Robustness experiments
- Error analysis

Dataset source:

https://data.mendeley.com/datasets/c6hf7g5f3t/3

---

### 2. BanFakeNews-2.0

Used as an **independent external dataset** for cross-dataset generalization.

The downloaded version contains the official:

```text
train_cleaned.csv
val_cleaned.csv
test_cleaned.csv
```

The original files contain four internal labels. For binary misinformation detection, this project preserves the original value in `Original_Label` and maps the labels as:

```text
Original Label 0 -> Fake
Original Label 1 -> Fake
Original Label 2 -> Fake
Original Label 3 -> Real
```

The final binary representation is:

```text
0 = Fake
1 = Real
```

For the external-transfer experiment, the project uses the official `test_cleaned.csv` split as the independent evaluation set.

Dataset source:

https://data.mendeley.com/datasets/kjh887ct4j/1

---

## Models

The project compares classical NLP models with Transformer-based language models.

### Classical Baselines

- TF-IDF + Logistic Regression
- Word + Character TF-IDF + Linear SVM

The Linear SVM baseline combines:

- Word-level TF-IDF features
- Character-level TF-IDF features

This provides a strong non-Transformer reference model.

### Transformer Models

- BanglaBERT
- BanglaBERT-small
- XLM-RoBERTa
- Multilingual BERT

Main Bangla model:

```text
csebuetnlp/banglabert
```

Lightweight model used for repeated held-out experiments:

```text
csebuetnlp/banglabert_small
```

---

## Experimental Design

### 1. Standard In-Domain Evaluation

The primary dataset is divided into:

```text
70% Training
15% Validation
15% Testing
```

The split is stratified by class.

Exact duplicate texts are removed before evaluation to reduce leakage.

---

### 2. Topic Generalization

The model is trained on all available topics except one.

Example:

```text
Train:
Politics
Sports
Entertainment
Technology
...

Test:
Held-out topic
```

The experiment is repeated across eligible topics.

This evaluates whether the model learns misinformation-related language patterns or becomes overly dependent on topic-specific characteristics.

---

### 3. Source / Publisher Generalization

The model is evaluated on publisher domains that are completely absent from training.

Because some source domains are strongly associated with a specific class, the project uses paired held-out source folds to preserve both Fake and Real examples in the test set.

This experiment is particularly useful for identifying possible:

- Source-label dependency
- Dataset shortcuts
- Publisher-specific linguistic patterns

---

### 4. Temporal Generalization

The temporal experiment evaluates whether a model trained on earlier articles generalizes to newer articles.

The preferred split is:

```text
Training:   earlier years
Validation: later intermediate year
Testing:    future year
```

This is more realistic than randomly mixing older and newer articles across all splits.

---

### 5. Cross-Dataset Generalization

A model trained only on BanglaFakeNews2025 is evaluated on BanFakeNews-2.0.

No BanFakeNews-2.0 examples are used for primary model fitting.

This measures how much performance changes when the model encounters a separately collected dataset with different linguistic and source characteristics.

---

### 6. Noise Robustness

The project evaluates model sensitivity to noisy Bangla text.

Perturbations include:

- Character deletion
- Character swapping
- Word dropout
- Punctuation removal
- Whitespace changes
- Combined perturbations

The robustness gap is measured as:

```text
Robustness Drop = Clean Macro-F1 - Perturbed Macro-F1
```

---

### 7. Error Analysis

The pipeline saves high-confidence incorrect predictions for manual inspection.

The analysis helps identify:

- Shortcut learning
- Difficult linguistic constructions
- Topic-specific failures
- Source-specific failures
- Confident false predictions

---

## Evaluation Metrics

The project reports:

- Accuracy
- Macro Precision
- Macro Recall
- Macro F1
- Fake-class F1
- Real-class F1
- ROC-AUC
- Confusion Matrix

**Macro-F1** is treated as the primary metric because it gives equal importance to both classes.

Additional statistical analysis includes:

- Bootstrap confidence intervals
- Exact McNemar test for paired model comparison

---

## Initial Validation Results

The following values came from a reduced **QUICK_MODE validation run** used to verify the full pipeline. They demonstrate that the implementation works, but they should not be treated as final benchmark results.

| Experiment | Model | Macro-F1 |
|---|---|---:|
| Standard in-domain | BanglaBERT | ~0.980 |
| Standard in-domain | TF-IDF + Linear SVM | ~0.968 |
| Temporal generalization | BanglaBERT | ~0.962 |
| Cross-dataset transfer | BanglaBERT | ~0.755 |
| Strong text-noise robustness | BanglaBERT | ~0.925 |
| Source-held-out | BanglaBERT-small | ~0.26–0.31 |

### Initial Observation

The validation run suggests an important pattern:

> Very high in-domain classification performance does not necessarily translate to equally strong performance under source or cross-dataset distribution shift.

The large source-shift degradation must be interpreted carefully because the primary dataset contains strong source-label associations. This may indicate that models are learning source-related shortcuts in addition to misinformation-related language patterns.

---

## Repository Structure

```text
bangla-misinformation-generalization/
|
|-- README.md
|-- requirements.txt
|
|-- notebooks/
|   `-- Bangla_Misinformation_Generalization.ipynb
|
|-- results/
|   |-- standard_baseline_metrics.csv
|   |-- standard_transformer_metrics.csv
|   |-- topic_shift_baseline.csv
|   |-- topic_shift_transformer.csv
|   |-- source_shift_baseline.csv
|   |-- source_shift_transformer.csv
|   |-- temporal_shift_results.csv
|   |-- cross_dataset_results.csv
|   |-- robustness_results.csv
|   `-- generalization_gap_summary.csv
|
`-- figures/
    |-- fig_class_distribution.png
    |-- fig_topic_distribution.png
    |-- fig_domain_distribution.png
    |-- fig_confusion_svm_indomain.png
    |-- fig_confusion_banglabert_indomain.png
    |-- fig_topic_shift_svm.png
    |-- fig_topic_shift_transformer.png
    |-- fig_noise_robustness.png
    `-- fig_year_distribution.png
```

The `results/` and `figures/` folders are optional for a minimal repository, but they make the project easier to review without rerunning the notebook.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/bangla-misinformation-generalization.git
cd bangla-misinformation-generalization
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Main dependencies include:

- Python
- PyTorch
- Transformers
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- SciPy
- SentencePiece
- Accelerate

A CUDA-enabled GPU is recommended for Transformer training.

---

## Running the Project

The notebook is designed to run on:

- Kaggle
- Google Colab
- A local machine with a compatible GPU

Open:

```text
notebooks/Bangla_Misinformation_Generalization.ipynb
```

### Recommended First Run

Use:

```python
QUICK_MODE = True
```

This performs a reduced run to validate:

- Dataset paths
- Label mappings
- Model downloads
- GPU setup
- Training pipeline
- Output generation

For a larger experiment:

```python
QUICK_MODE = False
```

---

## Kaggle Dataset Paths

The original development environment used Kaggle-mounted datasets.

Example structure:

```text
/kaggle/input/.../bangla-fake-news-2025/
/kaggle/input/.../banfakenews-2/
```

If you run the notebook in a different environment, update the dataset paths in the configuration section.

---

## Generated Outputs

The notebook automatically generates paper/project-ready outputs such as:

```text
dataset_summary.csv
standard_baseline_metrics.csv
standard_transformer_metrics.csv
topic_shift_baseline.csv
topic_shift_transformer.csv
source_shift_baseline.csv
source_shift_transformer.csv
temporal_shift_results.csv
cross_dataset_results.csv
robustness_results.csv
generalization_gap_summary.csv
high_confidence_errors.csv
```

It also generates figures for:

- Class distribution
- Topic distribution
- Domain distribution
- Year distribution
- Confusion matrices
- Topic-shift performance
- Noise robustness

The output folder can be compressed automatically into:

```text
Bangla_NLP_Paper_Ready_Outputs.zip
```

---

## Important Dataset Considerations

### Source-Label Confounding

Some publisher domains in the primary dataset are strongly associated with either Fake or Real articles.

Therefore, source-held-out performance should not be interpreted only as a measure of model quality.

It can also reveal:

- Dataset construction bias
- Publisher-label dependency
- Shortcut learning

This is one of the reasons the project evaluates multiple kinds of distribution shift instead of relying only on conventional random test accuracy.

### Temporal Metadata

Before using the dataset for publication-grade temporal analysis, publication dates should be carefully audited for consistency.

For this repository, the temporal experiment is used primarily as a project-level demonstration of chronological generalization.

---

## Why This Project Matters

A misinformation classifier may achieve excellent performance on a familiar benchmark while failing when deployed on:

- New publishers
- New topics
- Future data
- Independently collected datasets
- Noisy user-generated text

This project demonstrates why NLP systems should be evaluated beyond standard random test splits.

The goal is not only to maximize classification accuracy, but to understand **when and why model performance changes**.

---

## Skills Demonstrated

This project demonstrates practical experience with:

### Natural Language Processing
- Bangla text processing
- Text classification
- TF-IDF
- Character and word n-grams
- Transformer tokenization

### Machine Learning
- Logistic Regression
- Linear SVM
- Train/validation/test design
- Class-aware evaluation

### Deep Learning
- Transformer fine-tuning
- BanglaBERT
- XLM-R
- Multilingual BERT
- PyTorch

### Research-Oriented Evaluation
- Distribution shift
- Temporal generalization
- Topic generalization
- Source generalization
- Cross-dataset transfer
- Robustness testing
- Error analysis
- Statistical comparison

---

## Limitations

This project has several limitations:

1. Source and label are not fully independent in the primary dataset.
2. External datasets may use different collection and annotation procedures.
3. Text perturbations are synthetic approximations of real-world noise.
4. Dataset date metadata should be audited before publication-grade temporal claims.
5. Results from `QUICK_MODE` are intended for pipeline validation rather than final benchmarking.
6. Misinformation detection is a complex task and should not be treated as fully solved by binary text classification alone.

---

## Future Extensions

Possible extensions include:

- Full-scale evaluation with all Transformer models
- Bangla LLM evaluation
- Parameter-efficient fine-tuning
- Cross-lingual misinformation detection
- Bangla-English code-mixed misinformation
- Multimodal misinformation detection
- Explainability using Integrated Gradients or SHAP
- Adversarial robustness testing
- Source-balanced dataset construction
- Domain adaptation
- Calibration and uncertainty estimation

---

## Reproducibility

To reproduce the project:

1. Download the datasets from their original sources.
2. Install the packages listed in `requirements.txt`.
3. Update the dataset paths in the notebook.
4. Use a GPU-enabled environment.
5. Run the notebook from top to bottom.
6. Use `QUICK_MODE = True` first to verify the setup.

The datasets themselves are not redistributed in this repository.

---

## Disclaimer

This project is intended for research and educational purposes.

Model predictions should not be treated as definitive fact-checking decisions. Automated misinformation detection systems can make mistakes and should be used alongside human verification and reliable fact-checking processes.

---

## Acknowledgements

This project uses publicly available Bangla misinformation datasets and pretrained NLP models developed by the broader Bangla NLP research community.

Special acknowledgement is given to the creators of:

- BanglaFakeNews2025
- BanFakeNews-2.0
- BanglaBERT

Please cite the original dataset and model publications when using these resources in academic work.

---

## License

The source code in this repository can be released under the MIT License.

The datasets are governed by their respective original licenses and are **not included** in this repository.
