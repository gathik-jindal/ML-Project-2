# ML-Project-2: Machine Learning Classification Models

A comprehensive machine learning project comparing various classification algorithms on two distinct problems: **Binary Classification** (Loan Risk Prediction) and **Multi-class Classification** (Travel Spending Prediction).

## Table of Contents
- [Project Overview](#project-overview)
- [Datasets](#datasets)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Models Implemented](#models-implemented)
- [Usage](#usage)
- [Work Division](#work-division)

## Project Overview

This project applies multiple machine learning approaches to solve classification problems:

1. **Binary-Class Problem**: Predict loan risk (`RiskFlag`: 0 or 1) based on applicant profiles including financial information, employment status, and demographic data.

2. **Multi-class Problem**: Predict travel spending categories based on trip characteristics such as destination country, travel companions, activities, and accommodation details.

## Datasets

### Binary-Class Problem (Loan Risk Prediction)
- **Training data**: 204,277 samples with 17 features
- **Test data**: 51,070 samples
- **Target**: `RiskFlag` (Binary: 0 = Low Risk, 1 = High Risk)
- **Features include**: ApplicantYears, AnnualEarnings, RequestedSum, TrustMetric, WorkDuration, QualificationLevel, WorkCategory, RelationshipStatus, etc.

### Multi-class Problem (Travel Spending Prediction)
- **Target**: `spend_category` (Multiple classes: 0, 1, 2)
- **Features include**: country, age_group, travel_companions, main_activity, visit_purpose, accommodation details, trip duration, etc.

## Project Structure

```
ML-Project-2/
├── readme.md
├── requirements.txt
├── Binary-Class Problem/
│   ├── datasets/
│   │   ├── train.csv
│   │   ├── test.csv
│   │   └── preprocessed/
│   ├── preprocessing.ipynb
│   ├── probabilistic_models.py
│   ├── categorical_config.json
│   ├── Tree and Ensemble Model/
│   │   ├── decision_tree.py
│   │   ├── random_forest.py
│   │   ├── light_gbm.py
│   │   └── xgboost_model.py
│   ├── Support Vector Machine/
│   │   └── support_vector_machine.ipynb
│   ├── Nerual Networks/
│   │   └── nerual_networks.ipynb
│   ├── Gaussian Mixture Modelling/
│   │   └── gaussian_mixture_modelling.ipynb
│   └── Unsupervise classification/
│       └── unsupervised_classification.ipynb
└── Multi-class Problem/
    ├── datasets/
    │   ├── train.csv
    │   ├── test.csv
    │   └── preprocessed/
    ├── preprocessing.ipynb
    ├── travel_model_v2.py
    ├── categorical_config.json
    ├── Tree and Ensemble Model/
    │   ├── decision_tree.py
    │   ├── random_forest.py
    │   ├── light_gbm.py
    │   └── xgboost_model.py
    ├── Suppor Vector Machine/
    │   └── support_vector_machine.ipynb
    ├── Neural Networks/
    │   └── nerual_networks.ipynb
    ├── Gaussian Mixture Modelling/
    │   └── gaussian_mixture_modelling.ipynb
    └── Unsupervised classification/
        └── unsupervised_clustering.ipynb
```

## Installation

### Prerequisites
- Python 3.8 or higher

### Setup

1. Clone the repository:
```bash
git clone https://github.com/gathik-jindal/ML-Project-2.git
cd ML-Project-2
```

2. Create a virtual environment:
```bash
python -m venv venv
```

3. Activate the virtual environment:

   - **Windows**:
   ```bash
   venv\Scripts\activate
   ```

   - **macOS/Linux**:
   ```bash
   source venv/bin/activate
   ```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

## Models Implemented

### Supervised Learning

| Category | Models |
|----------|--------|
| **Tree-Based & Ensemble** | Decision Tree, Random Forest, XGBoost, LightGBM |
| **Probabilistic & Distance-Based** | Naive Bayes, K-Nearest Neighbors, Logistic Regression |
| **Advanced Methods** | Support Vector Machine (various kernels), Neural Networks |

### Unsupervised Learning

| Category | Models |
|----------|--------|
| **Clustering** | K-Means, Hierarchical Clustering, DBSCAN |
| **Probabilistic** | Gaussian Mixture Models |

## Usage

### Data Preprocessing
Run the preprocessing notebook to clean and prepare the data:
```bash
cd "Binary-Class Problem"
jupyter notebook preprocessing.ipynb
```

### Running Models

**Tree-Based Models (Python scripts)**:
```bash
cd "Binary-Class Problem/Tree and Ensemble Model"
python decision_tree.py
python random_forest.py
python light_gbm.py
python xgboost_model.py
```

**Probabilistic Models**:
```bash
cd "Binary-Class Problem"
python probabilistic_models.py
```

**Jupyter Notebooks**: Open the respective `.ipynb` files for SVM, Neural Networks, GMM, and Unsupervised models.

### Output
Each model generates a submission CSV file (e.g., `submission_decision_tree.csv`) containing predictions for the test dataset.

## Work Division

**Person A - Tree-Based and Ensemble Methods**:
- Decision Trees (baseline)
- Random Forest
- Gradient Boosting (XGBoost/LightGBM)
- Applied to both datasets

**Person B - Probabilistic and Distance-Based Methods**:
- Naive Bayes
- K-Nearest Neighbors
- Logistic Regression (for both binary and multi-class)
- Applied to both datasets

**Person C - Advanced Methods**:
- Support Vector Machines (with different kernels)
- Gaussian Mixture Models (for clustering comparison)
- Neural Networks
- Unsupervised clustering analysis (K-means, Hierarchical, DBSCAN)
- Applied to both datasets
