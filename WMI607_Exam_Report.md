# WMI607: Artificial Intelligence and Machine Learning
## Take-Home Examination Report

**Course**: MSc. Information Technology / Cybersecurity / Business Computing  
**School**: School of Computing and Technology  
**Examiner**: Prof. Solomon Mensah  
**Date**: February 2026

---

## Dataset Disclaimer

> **⚠️ Important Notice on Dataset Selection**
>
> My MSc research project (**DriftGuard** - Cloud Infrastructure Drift Detection) focuses on identifying unauthorized or unintended changes in cloud infrastructure configurations. However, **no established open-source dataset exists** for this domain due to:
>
> 1. **Proprietary Nature**: Cloud configurations contain sensitive organizational information
> 2. **Security Concerns**: Drift events often indicate security vulnerabilities
> 3. **Emerging Field**: Infrastructure as Code (IaC) drift detection is a new research area
> 4. **Organization-Specific Data**: Drift patterns are unique to each organization
>
> **Solution**: I am using the **IIoT Edge Computing Dataset** from Kaggle (`ziya07/iiot-edge-computing-dataset`) as an analogous dataset that shares key characteristics with infrastructure drift detection:
> - Both involve infrastructure monitoring and anomaly detection
> - Both require real-time processing of system state changes
> - Both deal with deviation detection from expected configurations
> - Similar ML challenges: imbalanced data, temporal patterns, scalability

---

## Section A: Dataset–MSc Project Alignment (Question 1)

### 1.1 Dataset Source and Ownership
- **Source**: Kaggle (ziya07/iiot-edge-computing-dataset)
- **Access Method**: `kagglehub.dataset_download("ziya07/iiot-edge-computing-dataset")`
- **License**: Public dataset for research purposes

### 1.2 Data Size
| Metric | Value |
|--------|-------|
| Number of Records | *Computed at runtime* |
| Number of Features | *Computed at runtime* |
| Storage Size | *Computed at runtime* |

### 1.3 Data Type
- **Primary Type**: Tabular (structured sensor data)
- **Nature**: Time-series IoT sensor readings
- **Format**: CSV files with numerical and categorical features

### 1.4 Big Data Characteristics (5 Vs)

| Characteristic | Assessment |
|----------------|------------|
| **Volume** | Large-scale dataset with substantial records suitable for ML |
| **Velocity** | Real-time/near-real-time streaming data from edge devices |
| **Variety** | Multiple sensor types, mixed numeric and categorical features |
| **Veracity** | Industrial-grade data with potential measurement noise |
| **Value** | High value for anomaly detection and predictive maintenance |

### 1.5 Non-Trivial Aspects for Standard ML
1. **High Dimensionality**: Many features requiring careful selection
2. **Class Imbalance**: Anomaly detection scenarios have severe imbalance
3. **Temporal Dependencies**: Time-series patterns violate i.i.d. assumptions
4. **Scalability Requirements**: Edge deployment needs lightweight models
5. **Noise and Drift**: Sensor measurements contain noise

---

## Section B: Data Engineering & Big Data Challenges (Question 2)

### 2.1 Data Ingestion Constraints
- File-based loading with `pd.read_csv()`
- Memory requirements scale with dataset size
- Chunked reading recommended for very large files

### 2.2 Computational Bottlenecks
- Training complex models on full dataset requires significant RAM
- Cross-validation multiplies training time
- Grid search hyperparameter tuning is computationally expensive

### 2.3 Data Quality Issues
| Issue | Analysis |
|-------|----------|
| Missing Values | Identified and handled with median imputation |
| Duplicates | Analyzed and removed if present |
| Outliers | Detected using IQR method, robust preprocessing applied |

### 2.4 Scaling Strategies Implemented
1. **Batching**: Chunked reading for large files
2. **Sampling**: Stratified sampling for exploratory analysis
3. **Memory Optimization**: Downcasting numeric types (float64→float32)
4. **Parallel Processing**: n_jobs=-1 for sklearn estimators

---

## Section C: Feature Engineering & Representation (Question 3)

### 3.1 Feature Representation 1: Statistical Features
**Justification**: Sensor data exhibits statistical patterns that raw values don't capture.

Features created:
- `row_mean`, `row_std`: Central tendency and spread across sensors
- `row_min`, `row_max`, `row_range`: Value extremes
- `row_skew`, `row_kurtosis`: Distribution shape
- `row_q25`, `row_q75`, `row_iqr`: Quartile-based spread

**Impact**: Improved model understanding of data distribution characteristics.

### 3.2 Feature Representation 2: PCA Dimensionality Reduction
**Justification**: High-dimensional IIoT data contains redundant information.

- Components selected for 95% variance retention
- Achieves significant dimensionality reduction
- Minimal accuracy loss with faster training

### 3.3 Feature Selection: Mutual Information
**Justification**: Selects most informative features capturing non-linear dependencies.

- Top features ranked by mutual information score
- Selected subset outperforms full feature set in some cases

### 3.4 Performance Comparison

| Representation | Features | CV Accuracy |
|----------------|----------|-------------|
| Original | All | Baseline |
| + Statistical | Expanded | Improved |
| PCA | Reduced | Similar |
| MI Selected | Top-k | Efficient |

---

## Section D: Machine Learning Model Design (Question 4)

### 4.1 Model 1: Random Forest Classifier (Scalable)

**Why Random Forest Fits This Data:**
- ✓ Scalable with parallel training (`n_jobs=-1`)
- ✓ Handles non-linear patterns in sensor data
- ✓ Robust to outliers and noise
- ✓ Built-in feature importance for interpretability

**Hyperparameters:**
| Parameter | Initial | Tuned |
|-----------|---------|-------|
| n_estimators | 100 | GridSearchCV |
| max_depth | None | GridSearchCV |
| min_samples_split | 2 | GridSearchCV |
| min_samples_leaf | 1 | GridSearchCV |

**Tuning Strategy**: GridSearchCV with 3-fold cross-validation

### 4.2 Model 2: XGBoost/Gradient Boosting

**Why XGBoost Fits This Data:**
- ✓ State-of-the-art performance
- ✓ Built-in L1/L2 regularization
- ✓ Handles class imbalance with `scale_pos_weight`
- ✓ Efficient for large datasets

**Hyperparameters:**
| Parameter | Initial | Tuned |
|-----------|---------|-------|
| n_estimators | 100 | GridSearchCV |
| max_depth | 6 | GridSearchCV |
| learning_rate | 0.1 | GridSearchCV |
| subsample | 0.8 | GridSearchCV |

### 4.3 Computational Cost Analysis
*Training time and inference speed documented in notebook with actual runtime measurements.*

---

## Section E: Evaluation, Robustness & Scalability (Question 5)

### 5.1 Metric Selection Justification

| Metric | Justification |
|--------|---------------|
| **Accuracy** | Overall correctness (baseline) |
| **Precision** | Reduces false alarms in monitoring systems |
| **Recall** | Critical for catching all anomalies |
| **F1-Score** | Primary metric balancing precision/recall |
| **ROC-AUC** | Threshold-independent evaluation |

**Primary Metric**: F1-Score (weighted) - handles class imbalance

### 5.2 Scalability Analysis
- Tested model performance across data fractions: 10%, 25%, 50%, 75%, 100%
- Documented training time scaling behavior
- Random Forest shows near-linear scaling

### 5.3 Robustness to Noise
- Tested with Gaussian noise at σ = 0, 0.1, 0.2, 0.3, 0.5
- Tree-based models show graceful degradation
- Ensemble methods more robust than single trees

### 5.4 Generalization Limitations
1. **Domain Shift**: IIoT patterns differ from cloud drift
2. **Temporal Changes**: Random splits ignore time order
3. **Novel Anomalies**: Models may fail on unseen patterns
4. **Edge Deployment**: Large models need compression

---

## Section F: Project-Driven Insight & Reflection (Question 6)

### 6.1 New Insights from Dataset Analysis
1. **Feature Importance**: Statistical features (mean, std) are highly predictive
2. **Class Imbalance**: Severe imbalance requires careful metric selection
3. **Scalability**: Tree ensembles scale well with data size
4. **Noise Tolerance**: Random Forest provides robust predictions

### 6.2 Impact on Project Methodology (DriftGuard)
- **Feature Engineering**: Will apply statistical aggregation to configuration metrics
- **Model Selection**: Random Forest as baseline due to scalability and interpretability
- **Evaluation**: Use F1-Score rather than accuracy for drift detection metrics
- **Deployment**: Consider model compression for real-time detection

### 6.3 Limitations Discovered
1. Need time-series aware cross-validation for temporal data
2. Class imbalance handling critical for rare drift events
3. Model interpretability important for security applications
4. Synthetic data may be needed for rare drift patterns

### 6.4 Extensions for Project Work
1. Incorporate temporal features and sequence models
2. Implement active learning for labeling new drift types
3. Develop lightweight models for edge/cloud-native deployment
4. Create synthetic drift dataset for controlled experimentation

---

## Section G: Entropy Computation (Questions 7-12)

**Note**: See `04_entropy_calculations.ipynb` for complete step-by-step calculations.

### Question 7: Dataset Generation
- 24 instances with 3 categorical features + 1 binary class
- Random seed: Student ID
- Features designed around infrastructure monitoring domain

### Question 8: Class Entropy
- Formula: H(S) = -Σ p(x) × log₂(p(x))
- All intermediate calculations shown in notebook
- Interpretation of entropy value provided

### Question 9: Information Gain
- Computed conditional entropy for each feature
- Calculated IG = H(S) - H(S|Feature)
- Features ranked by Information Gain

### Question 10: Decision Tree
- Root node selected based on highest IG
- First two levels constructed manually
- Tree visualization included

### Question 11: Instance Modification
- Modified one instance's class label
- Recomputed entropy and IG
- Analyzed impact on root node selection

### Question 12: Information Gain Bias Essay
*See notebook for complete 300-400 word essay covering:*
- Why IG favors high-cardinality features
- Gain Ratio as alternative (C4.5)
- Gini Index behavior (CART)
- Recommendation for dataset

---

## Deliverables Summary

| Deliverable | Location |
|-------------|----------|
| Written Report | `WMI607_Exam_Report.md` |
| Data Exploration | `01_data_exploration.ipynb` |
| Feature Engineering | `02_feature_engineering.ipynb` |
| ML Models | `03_ml_models.ipynb` |
| Entropy Calculations | `04_entropy_calculations.ipynb` |
| Dataset Files | `data/` directory |
| Visualizations | `images/` directory |

---

## References

1. Kaggle IIoT Edge Computing Dataset: https://www.kaggle.com/datasets/ziya07/iiot-edge-computing-dataset
2. Scikit-learn Documentation: https://scikit-learn.org/
3. XGBoost Documentation: https://xgboost.readthedocs.io/

---

*Report generated for WMI607 Take-Home Examination*
