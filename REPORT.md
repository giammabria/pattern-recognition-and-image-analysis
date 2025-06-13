# Machine Learning Methods for Pattern Recognition and Image Analysis

## Introduction

This report summarizes the methodologies, preprocessing techniques, validations, and comparative evaluations conducted during the four laboratory sessions for the course "Pattern Recognition and Image Analysis." The analysis was carried out using a subset of the CIFAR-10 dataset, consisting of three classes: 'airplane', 'bird', and 'horse'. Each image was represented by a 256-dimensional Histogram of Gradient (HoG) feature vector. The aim was to evaluate and compare various supervised learning algorithms.

---

## Lab 0: Introduction to the Dataset and PCA Preprocessing

### Dataset Overview

For these experiments, we used a modified CIFAR-10 dataset containing only three classes:

- **Class labels**: `['Airplane', 'Bird', 'Horse']`
- **Dataset structure**: 
  - `dataset.train` keys: `['images', 'hog', 'labels']`
  - **Training set**: 15,000 samples (5,000 per class)
  - **Test set**: 3,000 samples (1,000 per class)
  - **HoG features shape**: (15000, 256)

The class distribution is perfectly balanced, which simplifies interpretation and comparison of classification performance.

### Descriptive Data Analysis

#### Raw HoG Feature Distribution

We started with an examination of the raw HoG features. A bar plot of the HoG vector for the first training sample revealed the following:

- All HoG values are **positive and small**, centered near 0.
- The values span a range from near-zero to a maximum of approximately 0.0175.
- The dataset is **not centered around the origin**, which may hinder performance for classifiers that assume zero-mean data (e.g., SVMs, neural networks).

![HoG Sample 0 Vector](./e2cd12fa-4e17-41c5-b02a-bcb7f4767243.png)

#### Feature Distribution by Index

We then visualized the distributions for three sample HoG features: #0, #50, and #150. Across all three, the following characteristics were observed:

- Strong **skew towards 0**, with most values being very small.
- Long-tailed distribution, indicating sparse activation in gradient space.
- Implies that many HoG features may carry minimal information individually.

![HoG Feature Histograms](./79038d4c-3863-42ab-a569-0239bf4ec723.png)

#### Global Feature Statistics

To get a sense of variability and bias across features:

- **Mean distribution** shows that most features average between 0.002 and 0.006.
- **Standard deviation distribution** indicates moderate spread, with most features below 0.01.

This confirms the need for **standardization (mean 0, variance 1)** prior to applying many ML methods.

![Mean and Std of HoG Features](./832ce774-7a89-49c5-a810-d4549391c3b6.png)

#### Feature Correlation

A correlation heatmap across the 256 HoG features revealed:

- Some **visible block-structured correlation patterns**, likely arising from spatial contiguity in the HoG descriptor.
- However, many features remain relatively uncorrelated, which is promising for dimensionality reduction techniques like PCA.

![HoG Correlation Matrix](./948b3512-209f-4b14-b78a-d20e3543fe8c.png)

### PCA for Dimensionality Reduction

To better understand the intrinsic dimensionality of the data, we applied **Principal Component Analysis (PCA)**:

- PCA was used to **reduce to 2D** for visualization.
- While it helped visualize class clusters to some extent, **significant overlap remained**, suggesting the classes are not linearly separable in the top-variance dimensions.
- Nevertheless, PCA helps to:
  - Remove redundancy.
  - Reduce computational cost.
  - Improve performance for some algorithms (e.g., SVM with linear kernel).

### Summary and Justification for Preprocessing

- **Standardization** is essential before applying most ML models to ensure optimal convergence and fairness in feature weighting.
- **PCA**, while not perfect for classification, may still assist in reducing noise and improving generalization, especially when combined with models that benefit from decorrelated input (e.g., SVM).


---

## Lab 1: k-Nearest Neighbors (k-NN) and Decision Trees

### Methodologies:
- **k-NN Classifier**: Used L2 distance with varying values of `k`.
- **Decision Tree**: Built based on entropy and information gain.

### Preprocessing:
- **Normalization**: HoG features were normalized to zero mean and unit variance for k-NN.
- No explicit feature engineering for Decision Trees.

### Validation:
- **Holdout validation** (train/test split).
- **Cross-validation** was used to tune `k` for k-NN.

### Results:
- k-NN: Accuracy decreased for large `k` due to oversmoothing; small `k` led to overfitting.
- Decision Trees: Performance improved with pruning to reduce overfitting.

### Insights:
- Normalization critical for distance-based methods (k-NN).
- Decision Trees require fewer assumptions but may overfit without pruning.
- k-NN sensitive to noisy data and dimensionality.

---

## Lab 2: Neural Networks

### Methodologies:
- Multi-layer Perceptron (MLP) with one hidden layer (ReLU activation).
- Trained using backpropagation and SGD.

### Preprocessing:
- Input normalization applied.
- No PCA, as neural networks can internally learn representations.

### Validation:
- Performance tracked using training/test split.
- Accuracy and loss curves used to diagnose under/overfitting.

### Results:
- Good performance with appropriate regularization (dropout).
- Network capacity directly impacted accuracy.
- High variance without regularization.

### Insights:
- Neural Networks handle high-dimensional data well.
- Require hyperparameter tuning (e.g., hidden size, learning rate).
- Sensitive to training size and initialization.

---

## Lab 3: Support Vector Machines (SVM)

### Methodologies:
- Linear and RBF kernel SVMs evaluated.
- Grid search for hyperparameters (`C`, `gamma`).

### Preprocessing:
- Standardization of HoG features crucial.
- PCA tested for dimensionality reduction, but full HoG outperformed PCA-reduced input.

### Validation:
- Grid search with cross-validation.
- Accuracy compared across kernels and parameter settings.

### Results:
- RBF kernel outperformed linear SVM, capturing non-linear boundaries.
- Best performance when using full-dimensional HoG with RBF and tuned parameters.

### Insights:
- SVMs robust and effective for small-medium datasets.
- RBF kernel well-suited for complex class boundaries.
- PCA hurt performance slightly due to information loss.

---

## Comparative Evaluation

| Method           | Preprocessing        | Accuracy | Notes |
|------------------|----------------------|----------|-------|
| k-NN             | Normalization        | Moderate | Fast, simple, but sensitive to `k` |
| Decision Tree    | None                 | Low-Moderate | Prone to overfitting, interpretable |
| Neural Network   | Normalization        | High     | Needs tuning and regularization |
| SVM (RBF)        | Normalization        | Very High| Best performer with tuning |

**Why Some Methods Perform Better:**
- SVM with RBF kernel handles non-linear boundaries effectively.
- Neural networks excel with sufficient data and regularization.
- Simpler methods like k-NN and Decision Trees suffer from high variance and bias, respectively.

---

## Recommendations for Improvement

- **Feature Engineering**: Explore additional descriptors beyond HoG, such as color histograms or deep features.
- **Data Augmentation**: Enrich training data with image transformations to improve generalization.
- **Ensemble Methods**: Combine classifiers (e.g., bagging for trees, voting ensembles).
- **Hyperparameter Optimization**: Automated tuning (e.g., grid/random search, Bayesian optimization).
- **Model Calibration**: Apply softmax or Platt scaling for probability estimates.

---

## Conclusion

Each classifier demonstrates unique strengths and limitations in the context of pattern recognition. SVMs and neural networks clearly outperform simpler models due to their ability to model non-linear patterns in high-dimensional feature spaces. Preprocessing like normalization and validation strategies like cross-validation are critical for reliable evaluation. With additional improvements such as data augmentation and feature expansion, performance could be further enhanced.
