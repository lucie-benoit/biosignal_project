from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping


def train(model, X, y):
    """
    Train the model on the provided data.
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(X_train, y_train, epochs=50, batch_size=30,
                        validation_data=(X_val, y_val), callbacks=[early_stopping])

    return history, X_val, y_val