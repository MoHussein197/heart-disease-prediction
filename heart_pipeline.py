# Heart Disease ML Pipeline

import joblib
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE, SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, classification_report, confusion_matrix
)
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# ===============================
# 2.1 Data Preprocessing & Cleaning 
# ===============================

# Fetch dataset 
heart_disease = fetch_ucirepo(id=45) 
X = heart_disease.data.features 
y = heart_disease.data.targets 
y = (y > 0).astype(int)
y = (y > 0).astype(int).squeeze()

print("Dataset shape: ", X.shape) 
print("Target distribution:\n", y.value_counts())

# Handle missing values
X = X.dropna()
y = y.loc[X.index]

# One-hot Data encoding
X = pd.get_dummies(X, columns=['cp', 'restecg', 'slope', 'thal'], drop_first=True)

# Ensure other categoricals are numeric
X['sex'] = X['sex'].astype(int)
X['fbs'] = X['fbs'].astype(int)
X['exang'] = X['exang'].astype(int)
X['ca'] = X['ca'].astype(int)

# MinMaxScaler
hd_scaler = MinMaxScaler()
hd_minmax = pd.DataFrame(hd_scaler.fit_transform(X), columns=X.columns)

# StandardScaler
std_scaler = StandardScaler()
hd_standard = pd.DataFrame(std_scaler.fit_transform(X), columns=X.columns)

# EDA
X['age'].hist(bins=20, edgecolor= 'black')
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

#BoxPlot
df = X.copy()
df['target'] = y.values

plt.figure(figsize=(8,5))
sns.boxplot(x="target", y="chol", data=df)
plt.title("Cholesterol by Heart Disease")
plt.xlabel("Heart Disease (0=No, 1=Yes)")
plt.ylabel("Cholesterol")
plt.show()

# Correlation heatmap
plt.figure(figsize=(10,8))
sns.heatmap(X.corr(), annot=False, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# ===============================
#2.2 Dimensionality Reduction - PCA
# ===============================

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=None)
pca_result = pca.fit_transform(scaled_data)

# Explained variance
explained_var = np.cumsum(pca.explained_variance_ratio_)
plt.plot(range(1, len(explained_var)+1), explained_var, marker='o')
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA - Explained Variance")
plt.grid()
plt.show()

# Take first 2 PCs for visualization
pca_2d = PCA(n_components=2).fit_transform(scaled_data)
plt.scatter(pca_2d[:,0], pca_2d[:,1], c=y.values.ravel(), cmap="coolwarm", alpha=0.7)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA Projection")
plt.show()

# ===============================
#2.3 Feature Selection 
# ===============================

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)
importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(10,5))
importances.head(10).plot(kind="barh")
plt.title("Top 10 Feature Importances (RF)")
plt.show()


# Use Logistic Regression (or RF) as base model
model = LogisticRegression(max_iter=1000)

# Apply RFE to select top 8 features
rfe = RFE(estimator=LogisticRegression(max_iter=1000), n_features_to_select=8)
rfe.fit(X, y.values.ravel())
selected_features_rfe = X.columns[rfe.support_]

# Apply Chi-Square Test
chi2_selector = SelectKBest(score_func=chi2, k=8)
chi2_selector.fit(X, y)
selected_chi2 = X.columns[chi2_selector.get_support()]

final_features = list(set(selected_features_rfe) | set(selected_chi2))
print("Final Feature Set:", final_features)

X_selected = X[final_features]

# ===============================
#2.4 Supervised Learning
# ===============================


X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=3, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel="linear", probability=True, random_state=42)
}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    print(f"\n=== {name} ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1:", f1_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

# confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - {name}")
plt.show()

if hasattr(model, "predict_proba"):
    y_proba = model.predict_proba(X_test_scaled)[:,1]
    auc = roc_auc_score(y_test, y_proba)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.2f})")
    plt.plot([0,1],[0,1], "k--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.title(f"ROC Curve - {name}")
    plt.show()

# decision tree model
dt_model = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_model.fit(X_train, y_train)

# predictions
y_pred = dt_model.predict(X_test)

# confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
plt.title("Confusion Matrix - Decision Tree")
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.show()

#Make logistic regression
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train_scaled, y_train)

y_pred_scaled = lr_model.predict(X_test_scaled)

#plot confusion matrix 
cm = confusion_matrix(y_test, y_pred_scaled)
sns.heatmap(cm, annot= True, fmt='d')
plt.title('Confusion matrix - Logistic regression')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# AUC score
y_proba = rf_model.predict_proba(X_test_scaled)[:, 1]
auc = roc_auc_score(y_test, y_proba)
print("AUC Score:", auc)

# ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba)

plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
plt.plot([0,1], [0,1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Random Forest")
plt.legend()
plt.show()

# ===============================
# 2.5 Unsupervised Learning
# ===============================

# Use scaled data
X_scaled = StandardScaler().fit_transform(X)

# KMeans with elbow method
inertia = []
K = range(1,10)
for k in K:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

plt.plot(K, inertia, marker="o")
plt.xlabel("Number of clusters")
plt.ylabel("Inertia")
plt.title("Elbow Method - KMeans")
plt.show()

#KMeans
kmeans = KMeans(n_clusters =2, random_state = 42)
clusters_kmeans = kmeans.fit_predict(X_scaled)
print("KMeans vs Actual:\n", pd.crosstab(clusters_kmeans, y.values.ravel()))

# Perform hierarchical clustering
Z = linkage(X_scaled, method='ward')

# Plot dendrogram
plt.figure(figsize=(8, 5))
dendrogram(Z)
plt.title("Hierarchical Clustering Dendrogram")
plt.show()

agg = AgglomerativeClustering(n_clusters=2, linkage="ward")
clusters_agg = agg.fit_predict(X_scaled)
print("Agglomerative vs Actual:\n", pd.crosstab(clusters_agg, y.values.ravel()))

# ===============================
# Save evaluation metrics
# ===============================

import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Ensure results folder exists
os.makedirs("results", exist_ok=True)

# Use scaled test data for all models that need it
model_predictions = {
    "Logistic Regression": (y_pred_scaled, lr_model.predict_proba(X_test_scaled)[:,1]),
    "Decision Tree": (y_pred, None),  # Decision tree doesn't support predict_proba by default
    "Random Forest": (rf_model.predict(X_test_scaled), rf_model.predict_proba(X_test_scaled)[:,1]),
    "SVM": (models["SVM"].predict(X_test_scaled), models["SVM"].predict_proba(X_test_scaled)[:,1])
}

# Save evaluation metrics
metrics_file = "results/evaluation_metrics.txt"
with open(metrics_file, "w") as f:
    for model_name, (y_pred_model, y_proba_model) in model_predictions.items():
        f.write(f"Model: {model_name}\n")
        f.write(f"Accuracy: {accuracy_score(y_test, y_pred_model):.4f}\n")
        f.write(f"Precision: {precision_score(y_test, y_pred_model):.4f}\n")
        f.write(f"Recall: {recall_score(y_test, y_pred_model):.4f}\n")
        f.write(f"F1 Score: {f1_score(y_test, y_pred_model):.4f}\n")
        if y_proba_model is not None:
            f.write(f"AUC Score: {roc_auc_score(y_test, y_proba_model):.4f}\n")
        f.write("\n")

print(f"✅ Metrics successfully saved to {metrics_file}")


# ===============================
#2.6 Hyperparameter Tuning
# ===============================

# Baseline Random Forest
rf = RandomForestClassifier(random_state=42)

# Grid Search Parameters
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid = GridSearchCV(rf, param_grid, cv=5, scoring="accuracy", n_jobs=1)
grid.fit(X_train, y_train)
print("Best RF Params (Grid):", grid.best_params_)
print("Best RF Score (Grid):", grid.best_score_)

param_dist = {
    'n_estimators': np.arange(50, 300, 50),
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': np.arange(2, 10),
    'min_samples_leaf': np.arange(1, 5)
}

rand = RandomizedSearchCV(rf, param_dist, n_iter=20, cv=5, scoring="accuracy", random_state=42, n_jobs=1)

rand.fit(X_train, y_train)
print("Best RF Params (Random):", rand.best_params_)
print("Best RF Score (Random):", rand.best_score_)

final_model = RandomForestClassifier(**grid.best_params_, random_state=42)
final_model.fit(X_train, y_train)

# ===============================
# 2.7 Model Export & Deployment
# ===============================

joblib.dump(final_model, "heart_pipeline.pkl")

loaded_model = joblib.load("heart_pipeline.pkl")

pred = loaded_model.predict(X_test)


with open("heart_pipeline.pkl", "wb") as f:
    pickle.dump(final_model, f)

with open("heart_pipeline.pkl", "rb") as f:
    loaded_model = pickle.load(f)
