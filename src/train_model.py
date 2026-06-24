"""
train_model.py

Purpose:
    Build a transfer-learning image classifier using MobileNetV2 (pretrained
    on ImageNet) as the feature extractor, train a custom classification
    head, then fine-tune the top layers of the base model. Evaluate with
    accuracy, per-class precision/recall, and a confusion matrix. Save the
    trained model.

Run with:
    python src/train_model.py
"""

from __future__ import annotations

import os

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

import dataset as dataset_module
from dataset import get_datasets, get_augmentation_layer, IMG_SIZE
from tf_compat import require_tensorflow, tf


LIGHTWEIGHT = os.environ.get("HF_SPACE") == "1"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "leaf_disease_model.keras")
HISTORY_PLOT_PATH = os.path.join(PROJECT_ROOT, "images", "training_curves.png")
CONFUSION_MATRIX_PATH = os.path.join(PROJECT_ROOT, "images", "confusion_matrix.png")

EPOCHS_A = 3 if LIGHTWEIGHT else 8
EPOCHS_B = 2 if LIGHTWEIGHT else 6
BATCH_SIZE = 8 if LIGHTWEIGHT else 16


def build_model(num_classes: int, img_size: int = IMG_SIZE):
    """
    Build the model: augmentation -> preprocessing -> backbone -> head.

    This function tries to use MobileNetV2 pretrained on ImageNet (the
    standard transfer-learning approach for image classification). If the
    pretrained weights can't be downloaded (e.g. restricted network
    environment with no access to storage.googleapis.com), it falls back
    to a small CNN trained from scratch, so the pipeline still runs
    end-to-end.

    Why MobileNetV2 as the primary choice?
    - Small (14MB), fast, designed for mobile/edge deployment — appropriate
      for a project deployed in a Streamlit app, not a GPU server.
    - Pretrained on ImageNet (1.4M images, 1000 classes) — it already knows
      general visual features (edges, textures, shapes, colors) that
      transfer well to leaf images, even though "leaf disease" was never
      one of its original classes.

    Why freeze the base model initially (transfer learning path)?
    - The base model's weights encode general visual features learned from
      ImageNet. If we let them update immediately with random gradients from
      our untrained head, we'd destroy those useful pretrained weights
      ("catastrophic forgetting"). Training the head first with the base
      frozen lets the head learn to USE the existing features before we
      risk disturbing them.

    Returns:
        (model, base_model_or_None, used_transfer_learning: bool)
    """
    require_tensorflow()
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

    try:
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(img_size, img_size, 3),
            include_top=False,
            weights="imagenet",
        )
        base_model.trainable = False  # freeze for Phase A

        inputs = tf.keras.Input(shape=(img_size, img_size, 3))
        x = get_augmentation_layer()(inputs)
        x = preprocess_input(x)
        x = base_model(x, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

        model = tf.keras.Model(inputs, outputs)
        print("Using MobileNetV2 with ImageNet pretrained weights (transfer learning).")
        return model, base_model, True

    except Exception as e:
        print(f"\nCould not load pretrained MobileNetV2 weights ({type(e).__name__}: {e}).")
        print("Falling back to a small CNN trained from scratch.\n")
        print("NOTE: On a machine with normal internet access, this function will")
        print("successfully use MobileNetV2 + ImageNet weights (the intended,")
        print("recommended approach for this project).\n")

        inputs = tf.keras.Input(shape=(img_size, img_size, 3))
        x = get_augmentation_layer()(inputs)
        x = tf.keras.layers.Rescaling(1.0 / 255)(x)  # scale [0,255] -> [0,1]

        x = tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same")(x)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same")(x)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same")(x)
        x = tf.keras.layers.MaxPooling2D()(x)

        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(128, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

        model = tf.keras.Model(inputs, outputs)
        return model, None, False


def compile_model(model: tf.keras.Model, learning_rate: float):
    require_tensorflow()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )


def plot_training_history(history_a, history_b, save_path: str):
    """Plot accuracy and loss curves across both training phases."""
    acc = history_a.history["accuracy"] + history_b.history["accuracy"]
    val_acc = history_a.history["val_accuracy"] + history_b.history["val_accuracy"]
    loss = history_a.history["loss"] + history_b.history["loss"]
    val_loss = history_a.history["val_loss"] + history_b.history["val_loss"]

    fine_tune_start = len(history_a.history["accuracy"])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(acc, label="Train Accuracy")
    axes[0].plot(val_acc, label="Validation Accuracy")
    axes[0].axvline(fine_tune_start, color="gray", linestyle="--", label="Phase B starts")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(loss, label="Train Loss")
    axes[1].plot(val_loss, label="Validation Loss")
    axes[1].axvline(fine_tune_start, color="gray", linestyle="--", label="Phase B starts")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=120)
    plt.close()


def evaluate_model(model: tf.keras.Model, val_ds, class_names: list):
    """
    Evaluate using per-class precision/recall/F1 and a confusion matrix —
    NOT just overall accuracy, since per-class performance can hide a model
    that's great at "Healthy" but terrible at "Rust", which matters a lot
    in practice (missing a Rust diagnosis has real consequences).
    """
    y_true = []
    y_pred = []

    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=120)
    plt.close()


def run_training():
    require_tensorflow()
    dataset_module.BATCH_SIZE = BATCH_SIZE
    train_ds, val_ds, class_names = get_datasets()
    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}")

    model, base_model, used_transfer_learning = build_model(num_classes)

    # --- Phase A: train the classification head (base model frozen, if transfer learning) ---
    compile_model(model, learning_rate=1e-3)
    phase_a_label = "head (base model frozen)" if used_transfer_learning else "full model (no pretrained base)"
    print(f"\n=== Phase A: Training {phase_a_label} ===")
    history_a = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_A,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)],
    )

    if used_transfer_learning:
        # --- Phase B: unfreeze the top layers of the base model and fine-tune ---
        # Why only the TOP layers (not all of MobileNetV2)?
        # - Early layers learn generic features (edges, colors, textures) that
        #   are useful for almost any image task — we keep these frozen.
        # - Later layers learn more task-specific features — unfreezing these
        #   lets the model adapt them slightly to leaf images specifically.
        # Why a much lower learning rate (1e-5)?
        # - The base model's weights are already good (pretrained). Large
        #   updates would destroy that. A tiny learning rate makes small,
        #   careful adjustments — this is the core idea of "fine-tuning".
        base_model.trainable = True
        fine_tune_at = len(base_model.layers) - 30  # unfreeze last 30 layers
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False

        compile_model(model, learning_rate=1e-5)
        print(f"\n=== Phase B: Fine-tuning top {len(base_model.layers) - fine_tune_at} layers of MobileNetV2 ===")
        history_b = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS_B,
            callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)],
        )
    else:
        # No second phase for the from-scratch CNN — train a few more epochs
        # at the same learning rate instead, just so plot_training_history
        # (which expects two history objects) still works.
        print("\n=== Phase B: Additional training epochs (no separate fine-tuning phase) ===")
        history_b = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS_B,
            callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)],
        )

    # --- Evaluation ---
    print("\n=== Final Evaluation ===")
    evaluate_model(model, val_ds, class_names)
    plot_training_history(history_a, history_b, HISTORY_PLOT_PATH)

    # --- Save ---
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")
    print(f"Transfer learning used: {used_transfer_learning}")

    return model, class_names


if __name__ == "__main__":
    run_training()
