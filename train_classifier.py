"""
Train a linear probe on ViT features.
Handles empty classes and class imbalance automatically.
Outputs: linear_probe.joblib, style_names_clean.joblib, eval_report.txt
"""
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import json

# ── Load data ────────────────────────────────────────────────────────────────
X = np.load('features.npy')
y = np.load('labels.npy')
styles = joblib.load('style_names.joblib')

print(f"Raw data: {X.shape[0]} samples, {X.shape[1]} features, {len(styles)} classes")

# ── Drop empty classes ────────────────────────────────────────────────────────
valid_indices = []
valid_label_map = {}   # old_idx -> new_idx
clean_styles = []

new_idx = 0
for i, style in enumerate(styles):
    count = int((y == i).sum())
    if count > 0:
        valid_indices.append(i)
        valid_label_map[i] = new_idx
        clean_styles.append(style)
        new_idx += 1
    else:
        print(f"  Dropping '{style}' — 0 samples")

# Filter samples
mask = np.isin(y, valid_indices)
X_clean = X[mask]
y_clean = np.array([valid_label_map[label] for label in y[mask]])

print(f"\nClean data: {X_clean.shape[0]} samples, {len(clean_styles)} classes")
print("Class distribution:")
for i, s in enumerate(clean_styles):
    print(f"  [{i}] {s}: {int((y_clean == i).sum())}")

# ── Train/val split ───────────────────────────────────────────────────────────
X_train, X_val, y_train, y_val = train_test_split(
    X_clean, y_clean, test_size=0.2, random_state=42, stratify=y_clean
)
print(f"\nTrain: {len(X_train)}  Val: {len(X_val)}")

# ── Build pipeline: scaler + logistic regression ─────────────────────────────
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(
        max_iter=2000,
        solver='saga',
        class_weight='balanced',   # handles imbalance
        C=1.0,
        random_state=42
    ))
])

# ── Cross-validation ──────────────────────────────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipeline, X_clean, y_clean, cv=cv, scoring='accuracy')
print(f"\n5-fold CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ── Final fit on full train set ───────────────────────────────────────────────
pipeline.fit(X_train, y_train)
val_acc = pipeline.score(X_val, y_val)
print(f"Val accuracy:       {val_acc:.4f}")

# ── Detailed report ───────────────────────────────────────────────────────────
y_pred = pipeline.predict(X_val)
report = classification_report(y_val, y_pred, target_names=clean_styles, digits=4)
print(f"\nClassification Report:\n{report}")

cm = confusion_matrix(y_val, y_pred)
print("Confusion Matrix:")
print(cm)

# ── Also try SVM for comparison ───────────────────────────────────────────────
svm_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42))
])
svm_cv = cross_val_score(svm_pipeline, X_clean, y_clean, cv=cv, scoring='accuracy')
print(f"\nSVM 5-fold CV:      {svm_cv.mean():.4f} ± {svm_cv.std():.4f}")

# Pick best model
best_pipeline = pipeline if cv_scores.mean() >= svm_cv.mean() else svm_pipeline
best_name = "LogisticRegression" if cv_scores.mean() >= svm_cv.mean() else "SVM"
if best_name == "SVM":
    best_pipeline.fit(X_train, y_train)
print(f"\nSaving: {best_name}")

# ── Save artifacts ────────────────────────────────────────────────────────────
joblib.dump(best_pipeline, 'linear_probe.joblib')
joblib.dump(clean_styles, 'style_names_clean.joblib')

# Save label mapping for model.py
mapping = {
    'styles': clean_styles,
    'dropped': [s for s in styles if s not in clean_styles],
    'best_model': best_name,
    'cv_accuracy': float(cv_scores.mean()),
    'val_accuracy': float(val_acc)
}
with open('classifier_meta.json', 'w') as f:
    json.dump(mapping, f, indent=2)

# Save readable report
with open('eval_report.txt', 'w') as f:
    f.write(f"Model: {best_name}\n")
    f.write(f"CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n")
    f.write(f"Val Accuracy: {val_acc:.4f}\n\n")
    f.write(f"Classification Report:\n{report}\n")
    f.write(f"Confusion Matrix:\n{cm}\n")

print("\nSaved: linear_probe.joblib, style_names_clean.joblib, classifier_meta.json, eval_report.txt")
print("Done!")