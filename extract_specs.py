# Extract mel-spectrograms from config.AUDIO_PATH (one subfolder per species)
# into config.DATASET_PATH; low-signal chunks are partly kept in config.NOISE_PATH.
# Author: Stefan Kahl, 2018, Chemnitz University of Technology

import os

import numpy as np
import cv2

from sklearn.utils import shuffle

import config as cfg
from utils import audio
from utils import log

######################## CONFIG #########################
RANDOM = cfg.getRandomState()

######################### SPEC ##########################
def getSpecs(path):

    specs = []
    noise = []

    # Get mel-specs for file
    for spec in audio.specsFromFile(path,
                                    rate=cfg.SAMPLE_RATE,
                                    seconds=cfg.SPEC_LENGTH,
                                    overlap=cfg.SPEC_OVERLAP,
                                    minlen=cfg.SPEC_MINLEN,
                                    fmin=cfg.SPEC_FMIN,
                                    fmax=cfg.SPEC_FMAX,
                                    spec_type=cfg.SPEC_TYPE,
                                    shape=(cfg.IM_SIZE[1], cfg.IM_SIZE[0])):

        # Determine signal to noise ratio
        s2n = audio.signal2noise(spec)
        specs.append(spec)
        noise.append(s2n)

    # Shuffle arrays (we want to select randomly later)
    specs, noise = shuffle(specs, noise, random_state=RANDOM)

    return specs, noise

def parseDataset():

    # List of classes, subfolders as class names
    CLASSES = sorted(c for c in os.listdir(cfg.AUDIO_PATH) if os.path.isdir(os.path.join(cfg.AUDIO_PATH, c)))

    # Parse every class
    for c in CLASSES:

        print(c)

        # List all audio files
        afiles = sorted(os.listdir(os.path.join(cfg.AUDIO_PATH, c)))
        if not afiles:
            continue

        # Calculate maximum specs per file
        max_specs = cfg.MAX_SPECS_PER_CLASS // len(afiles) + 1

        # Get specs for every audio file
        for afile in afiles:

            spec_cnt = 0

            try:

                # Get specs and signal to noise ratios
                specs, noise = getSpecs(os.path.join(cfg.AUDIO_PATH, c, afile))

                # Save specs if it contains signal
                for s in range(len(specs)):

                    # NaN?
                    if np.isnan(noise[s]):
                        noise[s] = 0.0

                    # Above SIGNAL_THRESHOLD?
                    if noise[s] >= cfg.SPEC_SIGNAL_THRESHOLD:
                        filepath = os.path.join(cfg.DATASET_PATH, c)

                        # Count specs
                        spec_cnt += 1

                    # Rejected specs: we don't want to save every sample
                    elif RANDOM.choice([True, False], p=[cfg.NOISE_SAVE_PROBABILITY, 1 - cfg.NOISE_SAVE_PROBABILITY]):
                        filepath = cfg.NOISE_PATH

                    else:
                        filepath = None

                    if filepath:
                        os.makedirs(filepath, exist_ok=True)

                        # Filename contains s2n-ratio
                        filename = str(int(noise[s] * 10000)).zfill(4) + '_' + os.path.splitext(afile)[0] + '_' + str(s).zfill(3)

                        cv2.imwrite(os.path.join(filepath, filename + '.png'), specs[s] * 255.0)

                    # Do we have enough specs already?
                    if spec_cnt >= max_specs:
                        break

                # Stats
                log.i((spec_cnt, 'specs'))

            except Exception as e:
                log.e((spec_cnt, 'specs', 'ERROR DURING SPEC EXTRACT', afile, e))


if __name__ == '__main__':

    parseDataset()
