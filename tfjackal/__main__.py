""" """

import sys
import os

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

from importlib.resources import files

# import numpy as np
import tensorflow as tf
from . import plotter


from edi.schedulers.learningRateSchedulers import (  # pylint: disable=unused-import
    WarmUpCosine,
)

CIFAR10_TRAIN_WITH_VAL_SIZE = 40000


def get_cifar10_dataset(validation_split: bool = False):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    print(f"Training samples: {len(x_train)}")

    if validation_split:
        (x_train, y_train), (x_val, y_val) = (
            (
                x_train[:CIFAR10_TRAIN_WITH_VAL_SIZE],
                y_train[:CIFAR10_TRAIN_WITH_VAL_SIZE],
            ),
            (
                x_train[CIFAR10_TRAIN_WITH_VAL_SIZE:],
                y_train[CIFAR10_TRAIN_WITH_VAL_SIZE:],
            ),
        )

        print(f"Validation samples: {len(x_val)}")

    print(f"Testing samples: {len(x_test)}")

    if validation_split:
        return ((x_train, y_train), (x_val, y_val), (x_test, y_test))
    else:
        return ((x_train, y_train), (x_test, y_test))


def prepare_data(images, labels, batch_size: int | None = None):
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))

    if batch_size is not None:
        dataset = dataset.batch(batch_size)

    return dataset.prefetch(tf.data.AUTOTUNE)


def get_actual_and_predicted(dataset, model):
    actual = [labels for _, labels in dataset.unbatch()]
    predicted = model.predict(dataset)

    actual = tf.stack(actual, axis=0)
    predicted = tf.concat(predicted, axis=0)
    predicted = tf.argmax(predicted, axis=1)

    return actual, predicted


def create_full_model(augmentor, patch_layer, patch_encoder, classifier):
    inputs = tf.keras.layers.Input((32, 32, 3))
    x = augmentor(inputs)  # Shape(48, 48, 3)
    x = patch_layer(x)
    # TODO: Change when downstream no longer == true
    (x, _, _, _, _) = patch_encoder(x)  # unmasked_embeddings,
    # x = patch_encoder(x)  # unmasked_embeddings,
    outputs = classifier(x)

    return tf.keras.Model(inputs, outputs, name="full_model")


def main():
    models_path = files("tfjackal")
    # Classifier_51-78Acc_128D_Baseline_0-25_100_UnMasked_ADAM.keras
    # Classifier_49-6Acc_128D_Baseline_0-25_100_50Masked_ADAM.keras
    # Classifier_46-15Acc_128D_Baseline_0-25_100_75Masked_ADAM.keras

    encoder_filename = "New_Test_Baseline_0-25_100.keras"
    classifier_filename = (
        # Classifier_51-78Acc_128D_Baseline_0-25_100_UnMasked_ADAM.keras
        # "Classifier_49-6Acc_128D_Baseline_0-25_100_50Masked_ADAM.keras"
        "Classifier_46-15Acc_128D_Baseline_0-25_100_75Masked_ADAM.keras"
    )

    encoder_path = models_path.joinpath(encoder_filename)
    classifier_path = models_path.joinpath(classifier_filename)

    mae_model = tf.keras.models.load_model(encoder_path)
    classifier_model = tf.keras.models.load_model(classifier_path)

    train_augmentor = mae_model.train_aug_model
    test_augmentor = mae_model.test_aug_model

    patch_layer = mae_model.patch_layer
    patch_encoder = mae_model.patch_encoder

    patch_encoder.num_mask = 48
    patch_encoder.downstream = False #True
    encoder = mae_model.encoder

    test_model = create_full_model(
        test_augmentor, patch_layer, patch_encoder, classifier_model
    )

    # (train_data, val_data, test_data) = get_dataset()
    (train_data, test_data) = get_cifar10_dataset()

    BATCH_SIZE = 256

    # train_ds = prepare_data(
    #     train_data[0],
    #     train_data[1],
    #     # augmentation_model=train_augmentation_model,
    #     batch_size=BATCH_SIZE,
    # )

    # val_ds = prepare_data(
    #     val_data[0],
    #     val_data[1],
    #     augmentation_model=test_augmentation_model,
    #     is_train=False,
    # )
    test_ds = prepare_data(
        test_data[0],
        test_data[1],
        # augmentation_model=test_augmentation_model,
        batch_size=BATCH_SIZE,
    )

    actual, predicted = get_actual_and_predicted(test_ds, test_model)
    plotter.plot_confusion_matrix(
        actual,
        predicted,
        [
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ],
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())

    """

    Generate confusion matrix to see where its getting confused
    Can do top 1 top 2 top 3 


    """
