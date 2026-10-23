# All settings for spectrogram extraction and training.
# Spectrogram settings adapted from Stefan Kahl's BirdCLEF-Baseline (2018).
# Values below are the ones the 2022 thesis model was actually trained with.

import os

import numpy as np

BASE_PATH = os.path.dirname(os.path.abspath(__file__))

# Fixed random seed
RANDOM_SEED = 1337

def getRandomState():
    return np.random.RandomState(RANDOM_SEED)

########################  DATASET  ########################

# Input wav files, one subfolder per species
AUDIO_PATH = os.path.join(BASE_PATH, 'data', 'audio')

# Output spectrograms, one subfolder per species
DATASET_PATH = os.path.join(BASE_PATH, 'data', 'specs')

# Rejected (low signal) spectrograms, kept for future noise augmentation
NOISE_PATH = os.path.join(BASE_PATH, 'data', 'noise')

######################  SPECTROGRAMS  ######################

# Type of frequency scaling, mel-scale = 'melspec', linear scale = 'linear'
SPEC_TYPE = 'melspec'

# Sample rate for recordings, other sampling rates will force re-sampling
SAMPLE_RATE = 44100

# Min and max frequency for low and high pass
SPEC_FMIN = 500
SPEC_FMAX = 15000

# Length of chunks for spec generation, overlap of chunks and chunk min length (seconds)
SPEC_LENGTH = 1.0
SPEC_OVERLAP = 0.25
SPEC_MINLEN = 1.0

# STFT edge padding. Unlike the values above, this one is NOT known from the thesis run:
# librosa's default changed from 'reflect' to 'constant' in 0.9.0 (Feb 2022). 'reflect' matches
# the Android apps' JLibrosa (with the 2026-10-23 feature fixes); see docs/plan/2026-10-22-restructure.md
SPEC_PAD_MODE = 'reflect'

# Threshold for distinction between noise and signal
SPEC_SIGNAL_THRESHOLD = 0.0001

# Share of rejected chunks saved to NOISE_PATH
NOISE_SAVE_PROBABILITY = 0.1

# Limit the amount of specs per class when extracting spectrograms
MAX_SPECS_PER_CLASS = 2000

# Image size (width, height); the model input is (height, width, 1)
IM_SIZE = (256, 128)

#########################  MODEL  #########################

# Number of filters in each ResStack (one entry per stack)
FILTERS = [16, 32, 64, 128]

# Filter multiplier
RESNET_K = 2

# Number of ResBlocks in one ResStack
RESNET_N = 2

# Kernel size of every convolution ('same' padding)
KERNEL_SIZE = (3, 3)

#######################  TRAINING  ########################

# Folder under models/ where checkpoints, logs and the exported model are written
# (models/resnet_0919_test is the committed reference model; use a new name to keep it intact)
RUN_NAME = 'resnet_run'
MODEL_PATH = os.path.join(BASE_PATH, 'models', RUN_NAME)

# Prefix of the exported files (<prefix>_model.tflite, <prefix>_labels.txt)
EXPORT_NAME = 'resnet'

EPOCHS = 99
BATCH_SIZE = 32

# Share of DATASET_PATH used for validation
VAL_SPLIT = 0.2

# Any optimizer name understood by keras, e.g. 'nadam', 'adam'
OPTIMIZER = 'nadam'

# ReduceLROnPlateau on val_loss
LR_REDUCE_FACTOR = 0.2
LR_REDUCE_PATIENCE = 5
LR_MIN = 3e-4

# Epochs without val_loss improvement before stopping (None = disabled)
EARLY_STOPPING_PATIENCE = None

# Class weight smoothing, see utils.training.compute_class_weights
CLASS_WEIGHT_MU = 0.15

####################  LOG  ######################

# Options for log mode are 'all', 'info', 'progress', 'error', 'result'
LOG_MODE = 'all'
