from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam


def build_model(pooling='max'):
    """
    Build the 1D CNN architecture for AF detection (Hsieh et al., 2020).
    
    Parameters:
    pooling (str): 'max' or 'average' — determines pooling strategy.
    
    Returns:
    keras.Model: compiled model
    """
    Pool = layers.MaxPooling1D if pooling == 'max' else layers.AveragePooling1D

    model = models.Sequential([
        layers.Conv1D(32, 5, activation='relu', input_shape=(9000, 1)),
        layers.BatchNormalization(),
        Pool(2),

        layers.Conv1D(32, 5, activation='relu'),
        Pool(2),

        layers.Conv1D(64, 5, activation='relu'),
        Pool(2),

        layers.Conv1D(64, 5, activation='relu'),
        Pool(2),

        layers.Conv1D(128, 5, activation='relu'),
        Pool(2),

        layers.Conv1D(128, 5, activation='relu'),
        Pool(2),
        layers.Dropout(0.5),

        layers.Conv1D(256, 5, activation='relu'),
        Pool(2),

        layers.Conv1D(256, 5, activation='relu'),
        Pool(2),
        layers.Dropout(0.5),

        layers.Conv1D(512, 5, activation='relu'),
        Pool(2),
        layers.Dropout(0.5),

        layers.Conv1D(512, 5, activation='relu'),
        layers.Flatten(),

        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(32, activation='relu'),
        layers.Dense(4, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


if __name__ == '__main__':
    # Petit test rapide pour vérifier que l'architecture se construit sans erreur
    model = build_model(pooling='max')
    model.summary()