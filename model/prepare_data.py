import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder


def load_prepared_data(segments_path='../data/processed/segments.npy',
                        labels_path='../data/processed/labels.npy'):
    """
    Load preprocessed segments and labels, and prepare them for Keras.

    Returns:
    X (ndarray): shape (n_samples, 9000, 1)
    y (ndarray): one-hot encoded labels, shape (n_samples, 4)
    label_encoder (LabelEncoder): fitted encoder, useful to retrieve class names
    """
    segments = np.load(segments_path, allow_pickle=True)
    labels = np.load(labels_path, allow_pickle=True)

    X = np.stack(segments).reshape(-1, 9000, 1).astype('float32')

    le = LabelEncoder()
    y_int = le.fit_transform(labels)
    y = to_categorical(y_int)

    return X, y, le


if __name__ == '__main__':
    # Test for verify that it works
    X, y, le = load_prepared_data()
    print(X.shape, y.shape)
    print("Order of classes :", le.classes_)