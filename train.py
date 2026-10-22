# Train the ResNet on the spectrograms in config.DATASET_PATH and export it as TFLite.
# Re-running resumes from the newest checkpoint in <MODEL_PATH>/checkpoints/.

import os

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

import keras
import tensorflow as tf

import config as cfg
from model import build_resnet
from utils import training


def load_datasets():
    common = dict(validation_split=cfg.VAL_SPLIT,
                  color_mode='grayscale',
                  shuffle=True,
                  seed=cfg.RANDOM_SEED,
                  image_size=(cfg.IM_SIZE[1], cfg.IM_SIZE[0]),
                  batch_size=cfg.BATCH_SIZE)

    # Pixel values stay in [0, 255]: the Android app feeds the model the same range
    train_ds = keras.utils.image_dataset_from_directory(cfg.DATASET_PATH, subset='training', **common)
    val_ds = keras.utils.image_dataset_from_directory(cfg.DATASET_PATH, subset='validation', **common)

    return train_ds, val_ds


def build_callbacks(class_names, checkpoint_path):
    callbacks = [
        keras.callbacks.ModelCheckpoint(os.path.join(checkpoint_path, training.CHECKPOINT_PATTERN),
                                        save_freq='epoch',
                                        verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                          factor=cfg.LR_REDUCE_FACTOR,
                                          patience=cfg.LR_REDUCE_PATIENCE,
                                          min_lr=cfg.LR_MIN),
        keras.callbacks.CSVLogger(os.path.join(cfg.MODEL_PATH, 'training_log.csv'), append=True),
        training.ExportTFLite(class_names, cfg.MODEL_PATH, cfg.EXPORT_NAME),
    ]

    if cfg.EARLY_STOPPING_PATIENCE is not None:
        callbacks.append(keras.callbacks.EarlyStopping(monitor='val_loss', patience=cfg.EARLY_STOPPING_PATIENCE))

    return callbacks


def main():
    training.use_gpu()

    checkpoint_path = os.path.join(cfg.MODEL_PATH, 'checkpoints')
    os.makedirs(checkpoint_path, exist_ok=True)

    train_ds, val_ds = load_datasets()
    class_names = train_ds.class_names

    class_weights = training.compute_class_weights(train_ds.file_paths, class_names, cfg.CLASS_WEIGHT_MU)
    print(f'class_weights {class_weights}')

    # Resume from the newest checkpoint if there is one
    initial_epoch, last_checkpoint = training.find_last_checkpoint(checkpoint_path)
    if last_checkpoint:
        print(f'Resuming from {last_checkpoint}')
        model = keras.models.load_model(last_checkpoint)
    else:
        model = build_resnet(len(class_names))
        model.compile(optimizer=cfg.OPTIMIZER,
                      loss=keras.losses.SparseCategoricalCrossentropy(),
                      metrics=['accuracy'])
    model.summary()

    model.fit(train_ds.prefetch(tf.data.AUTOTUNE),
              epochs=cfg.EPOCHS,
              initial_epoch=initial_epoch,
              class_weight=class_weights,
              validation_data=val_ds,
              callbacks=build_callbacks(class_names, checkpoint_path))


if __name__ == '__main__':
    main()
