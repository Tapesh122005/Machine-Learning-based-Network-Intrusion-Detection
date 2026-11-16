# Machine Learning Based Network Intrusion Detection System (NSL-KDD)

A Python mini project that compares seven supervised machine learning classifiers for detecting network intrusions on the NSL-KDD benchmark dataset. Built for the **Cyber Security Lab (7CS4-22)**, B.Tech VII Semester, Dept. of CSE, Swami Keshvanand Institute of Technology, Management & Gramothan, Jaipur.

## Overview

Signature-based IDS tools cannot recognise new or modified attacks. This project trains and evaluates ML classifiers on labelled connection records so that unseen attacks can still be detected. A single script, `nids.py`, runs the complete workflow: loading, preprocessing, training, evaluation and plotting.

Two tasks are studied:

- **Binary detection:** Normal vs Attack
- **Five-class detection:** Normal, DoS, Probe, R2L, U2R

## Models compared

Random Forest, Support Vector Machine (RBF), Decision Tree, K-Nearest Neighbours, Logistic Regression, Gaussian Naive Bayes and Gradient Boosting (`HistGradientBoostingClassifier`).

## Dataset

[NSL-KDD](https://www.unb.ca/cic/datasets/nsl.html) (Tavallaee et al., 2009), a cleaned version of KDD Cup 1999.

- `KDDTrain+` : 125,973 records
- `KDDTest+` : 22,544 records (contains 17 attack types not seen in training)

Place both files in a `DataFolder/` directory.

## Pipeline

1. Load KDDTrain+ and KDDTest+ and map each attack name to its category
2. Standardise 38 numeric features (`StandardScaler`) and one-hot encode 3 categorical features (`OneHotEncoder`)
3. Train every model inside a scikit-learn `Pipeline` (no test-data leakage)
4. Evaluate: accuracy, precision, recall, F1-score, ROC-AUC, confusion matrices, training and prediction time
5. Save metrics tables, classification reports and figures to `results/`

## Results (KDDTest+)

| Task | Best model | Accuracy |
| --- | --- | --- |
| Binary | Gradient Boosting | 80.9% (precision 0.970, recall 0.686) |
| Five-class | Gradient Boosting | 80.1% |

Key findings:

- DoS and Probe attacks are detected well; **R2L and U2R are mostly missed** because they are rare in the training data and partly new in the test data.
- Every model has high precision but limited recall: few false alarms, but many missed attacks.
- Accuracy alone is misleading for intrusion detection. Per-class recall and confusion matrices matter more.

## Project structure

```
.
|-- nids.py                  main script
|-- DataFolder/
|   |-- KDDTrain+.txt
|   `-- KDDTest+.txt
`-- results/                 generated metrics, reports and figures
```

## Requirements

Python 3, pandas, NumPy, scikit-learn, Matplotlib, Seaborn.

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## Usage

```bash
# Binary task (Normal vs Attack)
python nids.py --data-dir DataFolder

# Five-class task
python nids.py --data-dir DataFolder --multiclass

# Run only selected models
python nids.py --data-dir DataFolder --models "Decision Tree,Naive Bayes"

# Other options
#   --svm-samples N   rows used to train the SVM (default 25000, 0 = all)
#   --out-dir PATH    output folder (default results)
#   --synthetic       synthetic demo data, pipeline test only
```

## Limitations

- Single train/test split; no cross-validation or hyper-parameter tuning
- SVM trained on a 25,000-row subsample for speed
- Works on pre-recorded records; no live traffic capture

## Future work

Class balancing (SMOTE, class weights), threshold tuning, feature selection, deep learning (LSTM, autoencoders), and newer datasets such as CIC-IDS2017 or UNSW-NB15.

## Ethical note

Only the public NSL-KDD dataset is used. No live network, system or person was scanned, attacked or monitored, and no malicious code was written or run.

## Author

Tapesh Kumar Vijay (Roll No. 23ESKCS830)
Guide: Vikash, Assistant Professor, Dept. of CSE, SKIT Jaipur
