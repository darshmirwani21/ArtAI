import numpy as np
import joblib

X = np.load('features.npy')
y = np.load('labels.npy')
styles = joblib.load('style_names.joblib')

print("features shape:", X.shape)
print("labels shape:", y.shape)
print("label distribution:", {s: int((y == i).sum()) for i, s in enumerate(styles)})
print("styles:", styles)