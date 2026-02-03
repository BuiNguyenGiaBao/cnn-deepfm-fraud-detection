# Hybrid CNN–DeepFM for Fraud Detection on Tabular Data

### An Empirical Study on the IEEE-CIS Fraud Detection Dataset

---

## Abstract

Fraud detection on large-scale tabular data poses significant challenges due to extreme class imbalance, high dimensionality, and complex feature interactions. This project proposes a **Hybrid CNN–DeepFM architecture for freud detection** that combines convolutional neural networks for representation learning with Deep Factorization Machines for feature interaction modeling. Using the IEEE-CIS Fraud Detection dataset, the proposed approach demonstrates strong discriminative performance, achieving a validation ROC–AUC of approximately **0.923**, while maintaining stable convergence and reasonable recall for the minority fraud class.

---

## 1. Introduction

Traditional machine learning models for fraud detection rely heavily on manual feature engineering and explicit interaction design. While tree-based methods (e.g., Gradient Boosting) perform well on tabular data, deep learning approaches offer the potential to automatically learn high-level representations and nonlinear feature relationships.

However, applying deep neural networks directly to tabular data remains challenging. This work explores a hybrid solution that:

1. Treats tabular features as structured sequences for convolutional processing.
2. Employs attention and bilinear pooling to capture second-order interactions.
3. Utilizes DeepFM to model both low-order and high-order feature relationships.

---

## 2. Dataset

### 2.1 IEEE-CIS Fraud Detection Dataset

The dataset consists of transactional and identity information collected from real-world e-commerce transactions. It includes:

* **Transaction features**: amounts, product codes, time-related variables.
* **Identity features**: device, browser, and anonymized identity attributes.
* **Label**: `isFraud` (binary classification).

Key characteristics:

* Highly imbalanced classes (fraud ≈ 3%).
* Large number of missing values.
* High-dimensional heterogeneous features.

---

## 3. Data Preprocessing

The preprocessing pipeline includes:

1. **Table merging**

   * `train_transaction` + `train_identity`
   * `test_transaction` + `test_identity`
     using `TransactionID`.

2. **Missing value handling**

   * Numerical features: median imputation (computed on training data).
   * Categorical features: filled with a constant token `"missing"`.

3. **Categorical encoding**

   * Label encoding applied jointly on training and test sets to avoid index mismatch.

4. **Feature alignment**

   * Train and test sets are aligned to ensure identical feature spaces.

5. **Output format**

   * Cleaned datasets are stored as:

     * `train_processed.csv`
     * `test_processed.csv`

---

## 4. Methodology

### 4.1 CNN-Based Feature Extraction

The first stage of the model learns dense representations from tabular inputs by:

* Projecting features into an embedding space.
* Applying 1D convolutional layers to capture local feature patterns.
* Using attention pooling to emphasize informative latent dimensions.
* Modeling pairwise feature interactions via **low-rank bilinear pooling**, implemented through element-wise (Hadamard) products.

This module serves as a **learnable feature extractor** for tabular data.

---

### 4.2 DeepFM for Relationship Learning

The extracted CNN embeddings are passed to a DeepFM module, which consists of:

* A linear component (first-order effects),
* A factorization machine component (second-order interactions),
* A deep neural network (higher-order nonlinear interactions).

This design allows the model to jointly learn simple and complex relationships without manual feature crossing.

---

### 4.3 Training Objective

* **Loss function**: Binary Cross-Entropy with Logits
* **Optimization**: Adam optimizer
* **Evaluation metrics**:

  * ROC–AUC (primary)
  * Precision, Recall, F1-score (secondary)

Accuracy is reported for completeness but not emphasized due to class imbalance.

---

## 5. Experimental Results

### 5.1 Quantitative Performance

* **Validation ROC–AUC**: ~0.923
* **Validation F1-score**: ~0.62 (fraud class)
* **Precision–Recall trade-off**: High precision with moderate recall, consistent with conservative fraud detection systems.

### 5.2 Training Dynamics

* Stable convergence without severe overfitting.
* Validation AUC saturates after approximately 40–50 epochs.
* Training loss continues to decrease, while validation loss remains stable.

---

## 6. Discussion

The results indicate that combining CNN-based representation learning with DeepFM interaction modeling is effective for large-scale tabular fraud detection. Compared to standalone deep models, the hybrid architecture:

* Improves interaction modeling without explicit graph construction.
* Avoids high-dimensional one-hot encodings.
* Remains computationally tractable for large datasets.

However, recall for the fraud class remains a challenge due to extreme imbalance, suggesting future work on cost-sensitive learning or focal loss variants.

---

## 7. Limitations and Future Work

Potential extensions include:

* Incorporating focal loss or class-weighted objectives.
* Threshold optimization for recall-oriented deployment.
* Ensembling with gradient boosting models.
* Feature selection to reduce dimensionality and training time.

---

## 8. Conclusion

This study demonstrates that a Hybrid CNN–DeepFM architecture can effectively model complex feature interactions in tabular fraud detection tasks. The approach achieves competitive performance on the IEEE-CIS dataset while maintaining architectural interpretability and training stability.

---

## References

* Guo, H., et al. (2017). *DeepFM: A Factorization-Machine based Neural Network for CTR Prediction.*
* Kaggle IEEE-CIS Fraud Detection Competition.
* Recent empirical studies on CNNs for tabular data.



