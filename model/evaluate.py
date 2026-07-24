import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt


def evaluation_metrics(y_true, y_pred, class_names=['A', 'N', 'O', '~']):
    """
    y_true, y_pred: class indices (not one-hot), e.g. from argmax
    """
    f1_per_class = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2, 3])
    avg_f1 = f1_per_class.mean()  # simple mean

    metrics = {
        'f1_per_class': dict(zip(class_names, f1_per_class)),
        'average_f1': avg_f1
    }
    return metrics


def plot_confusion_matrix(cm, class_names, save_path=None):
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(7, 6))
    plt.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Normalized Confusion Matrix')
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')

    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            plt.text(j, i, f'{cm_norm[i, j]:.2f}',
                     ha='center', va='center',
                     color='white' if cm_norm[i, j] > 0.5 else 'black')

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved : {save_path}")
    
    plt.show()


def k_fold_cross_validation(build_model_fn, X, y_onehot, k=5, epochs=30, batch_size=30):
    """
    build_model_fn: fonction without argument returns a new compiled Keras model
    y_onehot: labels in one-hot encoding
    """
    y_int = np.argmax(y_onehot, axis=1)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    fold_metrics = []
    all_cm = np.zeros((4, 4))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_int)):
        print(f"--- Fold {fold+1}/{k} ---")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y_onehot[train_idx], y_onehot[val_idx]

        # new model at each fold
        model = build_model_fn()

        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                  validation_data=(X_val, y_val), callbacks=[early_stopping], verbose=1)

        y_pred_proba = model.predict(X_val)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_true = np.argmax(y_val, axis=1)

        metrics = evaluation_metrics(y_true, y_pred)
        fold_metrics.append(metrics)
        print(f"F1 moyen fold {fold+1} : {metrics['average_f1']:.3f}")

        all_cm += confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])

    avg_f1_overall = np.mean([m['average_f1'] for m in fold_metrics])
    print(f"\nF1 moyen sur {k} folds : {avg_f1_overall:.3f}")

    return fold_metrics, all_cm

def plot_training_curves(history, title='Training History', save_path=None):
    """
    Plot training and validation accuracy and loss curves.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history['accuracy'], label='Train')
    axes[0].plot(history.history['val_accuracy'], label='Validation')
    axes[0].set_title(f'{title} — Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()

    axes[1].plot(history.history['loss'], label='Train')
    axes[1].plot(history.history['val_loss'], label='Validation')
    axes[1].set_title(f'{title} — Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure sauved : {save_path}")
    
    plt.show()

def show_misclassified_example(model, X_val, y_val, le, true_class='A', pred_class='O'):
    """
    Show an example of a misclassified signal from the validation set.  
    """
    y_pred = np.argmax(model.predict(X_val), axis=1)
    y_true = np.argmax(y_val, axis=1)
    
    true_idx = np.where(le.classes_ == true_class)[0][0]
    pred_idx = np.where(le.classes_ == pred_class)[0][0]
    
    mask = (y_true == true_idx) & (y_pred == pred_idx)
    misclassified_indices = np.where(mask)[0]
    
    if len(misclassified_indices) == 0:
        print(f"No example of {true_class} misclassified as {pred_class} found.")
        return
    
    idx = misclassified_indices[0]
    signal = X_val[idx].flatten()
    
    plt.figure(figsize=(14, 5))
    plt.plot(signal)
    plt.title(f'Example {true_class} misclassified as {pred_class}')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude (mV)')
    plt.savefig('../results/misclassified_example.png', dpi=300, bbox_inches='tight')
    plt.show()
