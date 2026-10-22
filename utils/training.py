# Helpers for train.py: GPU setup, class weights, checkpoint resume and TFLite export.

import math
import os
import re
from collections import Counter

import tensorflow as tf


def use_gpu():
    # Use the first GPU only, unless CUDA_VISIBLE_DEVICES is already set
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


def count_samples_per_class(file_paths):
    # The class name is the parent folder of each image
    return Counter(os.path.basename(os.path.dirname(f)) for f in file_paths)


def compute_class_weights(file_paths, class_names, mu=0.15):
    """
    Log-smoothed class weights (never below 1.0), keyed by the label index keras
    assigns, i.e. the position of the class in class_names.
    """
    counts = count_samples_per_class(file_paths)
    total = sum(counts.values())

    class_weight = {}
    for index, name in enumerate(class_names):
        # A class with no training samples is treated as having one
        score = math.log(mu * total / max(counts[name], 1))
        class_weight[index] = max(score, 1.0)

    return class_weight


CHECKPOINT_PATTERN = 'weights.{epoch:02d}.keras'

def find_last_checkpoint(checkpoint_path):
    """Return (epoch, path) of the newest checkpoint, or (0, None) if there is none."""
    if not os.path.isdir(checkpoint_path):
        return 0, None

    epochs = [int(m.group(1)) for f in os.listdir(checkpoint_path)
              if (m := re.fullmatch(r'weights\.(\d+)\.keras', f))]
    if not epochs:
        return 0, None

    epoch = max(epochs)
    return epoch, os.path.join(checkpoint_path, CHECKPOINT_PATTERN.format(epoch=epoch))


class ExportTFLite(tf.keras.callbacks.Callback):
    """Export the model as TFLite after every epoch, so the latest model is always usable."""

    def __init__(self, class_names, model_path, model_name):
        super().__init__()
        self.class_names = class_names
        self.model_path = model_path
        self.model_name = model_name

    def on_epoch_end(self, epoch, logs=None):
        save_tflite(self.model, self.class_names, self.model_path, self.model_name)


def save_tflite(model, class_names, model_path, model_name):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Write to a temp file first so an interrupted run never leaves a truncated model
    tflite_path = os.path.join(model_path, model_name + '_model.tflite')
    with open(tflite_path + '.tmp', 'wb') as f:
        f.write(tflite_model)
    os.replace(tflite_path + '.tmp', tflite_path)

    # Labels in the same order as the model outputs
    with open(os.path.join(model_path, model_name + '_labels.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(class_names))
