"""
dataset.py

Purpose:
    Build TensorFlow datasets (train and validation) from the images in
    data/raw/<ClassName>/*.jpg, and define the data augmentation pipeline.

    Keeping this in its own file means train_model.py and predict.py can
    both import the same IMG_SIZE / CLASS_NAMES constants, so training and
    inference always agree on image size and class label order.
"""

from __future__ import annotations

import os

from tf_compat import require_tensorflow, tf

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")

IMG_SIZE = 160       # MobileNetV2 supports 96-224px; 160 balances speed and accuracy
BATCH_SIZE = 16
SEED = 42

# Fixed class order — must match the order used at training time for
# predictions to map back to the correct label. We sort alphabetically
# because that's also the order image_dataset_from_directory uses by default.
CLASS_NAMES = sorted(os.listdir(DATA_DIR)) if os.path.exists(DATA_DIR) else \
    ["Healthy", "Powdery_Mildew", "Rust"]


def get_datasets(data_dir: str = DATA_DIR, img_size: int = IMG_SIZE, batch_size: int = BATCH_SIZE):
    """
    Load images from data_dir into train/validation tf.data.Dataset objects.

    Why an 80/20 split with a fixed seed?
    - 80/20 is a reasonable default for a small dataset (240 images total).
    - A fixed seed (validation_split + seed) ensures the SAME images end up
      in train vs validation every time we re-run this, which makes results
      reproducible and comparable across training runs.
    """
    require_tensorflow()
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=SEED,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        label_mode="categorical",  # one-hot labels, needed for softmax + categorical_crossentropy
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        label_mode="categorical",
    )

    class_names = train_ds.class_names

    # Cache + prefetch: speeds up training by overlapping data loading with
    # model execution on the GPU/CPU. Standard tf.data performance practice.
    train_ds = train_ds.cache().shuffle(200, seed=SEED).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, class_names


def get_augmentation_layer() -> tf.keras.Sequential:
    """
    Data augmentation applied ONLY to the training set, INSIDE the model
    (as Keras layers). This means:
    1. Augmentation happens on-the-fly during training — no need to
       generate and store extra image files.
    2. At inference time, these layers are inactive (Keras layers like
       RandomFlip only apply during training=True), so predictions on a
       single uploaded image are unaffected.

    Why these specific augmentations?
    - RandomFlip("horizontal"): a leaf photographed from a slightly
      different angle is still the same leaf — flipping teaches the model
      this invariance.
    - RandomRotation(0.1): real photos won't always be perfectly upright.
    - RandomZoom(0.1): simulates photos taken from slightly different distances.

    With only 80 images per class, this is especially important — it
    effectively creates many slightly different versions of each image,
    reducing overfitting.
    """
    require_tensorflow()
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
    ], name="data_augmentation")


if __name__ == "__main__":
    train_ds, val_ds, class_names = get_datasets()
    print("Class names:", class_names)
    print("Train batches:", tf.data.experimental.cardinality(train_ds).numpy())
    print("Val batches:", tf.data.experimental.cardinality(val_ds).numpy())

    for images, labels in train_ds.take(1):
        print("Image batch shape:", images.shape)
        print("Label batch shape:", labels.shape)
