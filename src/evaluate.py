"""
Evaluates the trained model on the held-out test set: classification report
and confusion matrix.
"""

import keras
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

TEST_DIR = "Data/test_clahe"
IMAGE_SIZE = (200, 200)
BATCH_SIZE = 16


def load_test_data():
    return keras.utils.image_dataset_from_directory(
        directory=TEST_DIR,
        label_mode="categorical",
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        shuffle=False,
    )


def evaluate(model_path="models/model.keras"):
    test_data = load_test_data()
    model = keras.models.load_model(model_path)

    test_pred = model.predict(test_data)
    test_true = np.concatenate([y for _, y in test_data], axis=0)

    test_pred = np.argmax(test_pred, axis=1)
    test_true = np.argmax(test_true, axis=1)

    class_names = test_data.class_names

    report = classification_report(test_true, test_pred, target_names=class_names)
    print(report)

    with open("results/classification_report.txt", "w") as f:
        f.write(report)

    conf_matrix = confusion_matrix(test_true, test_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=class_names)
    disp.plot()
    plt.title("Steel Defect Classifier - Confusion Matrix")
    plt.tight_layout()
    plt.savefig("results/confusion_matrix.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    evaluate()
