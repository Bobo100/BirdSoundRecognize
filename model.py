# ResNet-style CNN for mel-spectrogram classification.
# Ported from the Lasagne ResNet in Stefan Kahl's BirdCLEF-Baseline.

import keras
from keras import layers

import config as cfg


def residual_block(net_in, filters, kernel_size, stride=1, preactivated=True):
    net_pre = layers.BatchNormalization()(net_in)
    net_pre = layers.Activation('relu')(net_pre)

    # Preactivated shortcut?
    net_sc = net_pre if preactivated else net_in

    # First convolution (always has preactivated input)
    net = layers.Conv2D(filters,
                        kernel_size,
                        padding='same',
                        strides=stride,
                        kernel_initializer=keras.initializers.HeNormal(),
                        activation='relu')(net_pre)
    net = layers.BatchNormalization()(net)

    # Second convolution
    net = layers.Conv2D(filters,
                        kernel_size,
                        padding='same',
                        strides=1,
                        kernel_initializer=keras.initializers.HeNormal(),
                        activation=None)(net)

    # Project the shortcut when the number of channels or the spatial size changes
    if stride != 1 or net.shape[-1] != net_sc.shape[-1]:
        shortcut = layers.Conv2D(filters,
                                 kernel_size=(1, 1),
                                 padding='same',
                                 strides=stride,
                                 kernel_initializer=keras.initializers.HeNormal(),
                                 activation=None)(net_pre)
    else:
        shortcut = net_sc

    return layers.add([net, shortcut])


def build_resnet(nclass, input_shape=(cfg.IM_SIZE[1], cfg.IM_SIZE[0], 1)):
    """
    Build the ResNet used in the thesis.
    :param nclass: number of output classes
    :param input_shape: (height, width, channels); grayscale pixel values in [0, 255]
    :return: keras Model
    """
    keras.utils.set_random_seed(cfg.RANDOM_SEED)

    input_ = layers.Input(shape=input_shape)

    # First convolution
    net = layers.Conv2D(cfg.FILTERS[0],
                        kernel_size=cfg.KERNEL_SIZE,
                        padding='same',
                        kernel_initializer=keras.initializers.HeNormal())(input_)

    # Residual stacks; the first block of each stack halves the spatial size
    for filters in cfg.FILTERS:
        net = residual_block(net, filters * cfg.RESNET_K, cfg.KERNEL_SIZE, stride=2)
        for _ in range(1, cfg.RESNET_N):
            net = residual_block(net, filters * cfg.RESNET_K, cfg.KERNEL_SIZE, preactivated=False)

    # Post activation
    net = layers.BatchNormalization()(net)
    net = layers.Activation('relu')(net)

    # Pooling
    net = layers.GlobalAveragePooling2D()(net)

    # Classification layer
    net = layers.Dense(units=nclass, kernel_initializer=keras.initializers.HeNormal())(net)
    net = layers.Activation('softmax')(net)

    return keras.Model(inputs=input_, outputs=net)
