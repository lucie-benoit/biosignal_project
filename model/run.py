from prepare_data import load_prepared_data
from model import build_model
from train import train
from evaluate import evaluation_metrics, plot_confusion_matrix, plot_training_curves, show_misclassified_example
import numpy as np
from sklearn.metrics import confusion_matrix

# 1. Load data
X, y, le = load_prepared_data()
print(f"Data : {X.shape}, classes : {le.classes_}")

# 2. Train with a simple split
model = build_model(pooling='max')
history, X_val, y_val = train(model, X, y)

# 3. Plot training curves
plot_training_curves(history, title='Proposed-1 (Max Pooling)', save_path='../results/curve_proposed1.png')

# 4. Evaluate
y_pred = np.argmax(model.predict(X_val), axis=1)
y_true = np.argmax(y_val, axis=1)

metrics = evaluation_metrics(y_true, y_pred, class_names=le.classes_)
print(metrics)

cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
plot_confusion_matrix(cm, class_names=le.classes_, save_path='../results/confusion_matrix.png')

# 5. Show a misclassified AF example 
show_misclassified_example(model, X_val, y_val, le, true_class='A', pred_class='O')