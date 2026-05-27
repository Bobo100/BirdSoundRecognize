# 🐦 BirdSoundRecognize

> Deep learning pipeline for bird species identification from audio recordings — a master's thesis project using mel-spectrograms and a custom ResNet architecture.

<p align="center">
  <img src="image/model%20flow%20chart.png" alt="System flow chart" width="720">
</p>

**Companion mobile app:** [bird-sound-recognize-app](https://github.com/Bobo100/bird-sound-recognize-app) — runs the trained TFLite model on Android for real-time on-device inference.

---

## What it does

1. **Crawls** bird recordings from [xeno-canto](https://xeno-canto.org/).
2. **Converts** audio (mp3 → wav) and extracts **mel-spectrograms** as 256×128 single-channel images.
3. **Trains** a custom ResNet-style CNN (Keras / TensorFlow) to classify species from spectrograms.
4. **Exports** the model for deployment on a companion Android app.

## Tech stack

| Layer | Tool |
|---|---|
| Framework | TensorFlow / Keras |
| Architecture | Custom ResNet (configurable filters, kernel sizes, ResBlocks) |
| Audio features | Mel-spectrogram (`librosa`-style), 44.1 kHz, 500–15000 Hz |
| Data augmentation | Vertical roll + additive noise samples |
| Optimizer | Adam, cosine LR schedule (0.001 → 1e-6) |
| Training | 60 epochs, batch size 32, early stopping at 10 |
| Language | Python 3.7.10 |

Tested on Windows 10, i7-6700, GTX 1070, CUDA 8.0, cuDNN 7.5, 24GB RAM.

## Repository structure

```
.
├── config.py                   # All hyperparameters and paths (single source of truth)
├── spec.py                     # Extract mel-spectrograms from wav files
├── mytrain_resnet18_ver1.py    # Main training script (ResNet-18 variant)
├── model_details.py            # Model architecture helpers + GPU setup
├── dataset-download/
│   ├── mp3_to_wav.py           # Audio format conversion utility
│   ├── dataset_input_folder_name/      # Place raw wav files here, organized by species
│   └── dataset_output_folder_name/     # Generated spectrograms land here
├── utils/                      # Audio I/O, batch generation, metrics, logging
└── image/                      # README assets
```

## Quick start

### 1. Prepare the dataset

Organize your audio files by species (any number of classes):

```
dataset-download/dataset_input_folder_name/
├── species1/
│   ├── species1_1.wav
│   └── species1_2.wav
└── species2/
    ├── species2_1.wav
    └── species2_2.wav
```

If your files are mp3, convert them first:

```bash
cd dataset-download
python mp3_to_wav.py   # remember to edit the mp3 folder path inside
```

### 2. Configure paths

Open [`config.py`](./config.py) and adjust the dataset paths near the top (`TRAINSET_PATH`, `AUDIO_FOLDER`, `DATASET_PATH`).

### 3. Extract spectrograms

```bash
python spec.py
```

This generates mel-spectrograms (256×128) under `dataset_output_folder_name/`.

### 4. Train

```bash
python mytrain_resnet18_ver1.py
```

Snapshots are saved under `snapshots/` every epoch; final results land in `BirdCLEF_Logfile.txt`.

## How it works

1. **Audio → image**: each clip is split into 1.0s windows (0.25s overlap), converted to mel-spectrograms in the 500–15000 Hz band, normalized to a fixed 256×128 image.
2. **Noise vs signal**: chunks below a signal threshold are routed to a noise pool and used later for augmentation rather than discarded.
3. **Model**: a ResNet-style CNN built from configurable residual stacks. Filter sizes, kernel sizes, and stack depths are all hyperparameters in `config.py`.
4. **Output**: per-species probability distribution. The mobile companion app runs this model on-device as TFLite for real-time predictions.

## Status

This project was part of a master's thesis (2022). It is **not actively maintained**, but the pipeline is reproducible end-to-end on the environment listed above. Issues and questions are welcome — I'll respond when I can.

## Acknowledgements & references

This codebase adapts work from prior research:

- **ResNet pipeline & spectrogram extraction** — adapted from [Stefan Kahl's BirdCLEF-Baseline](https://github.com/kahst/BirdCLEF-Baseline) (originally Lasagne; rewritten here for TensorFlow / Keras).
- **EfficientNet variant** — inspired by [_Sound-based Bird Classification_](https://towardsdatascience.com/sound-based-bird-classification-965d0ecacb2b) by Magdalena Kortas.
- **Dataset** — [xeno-canto](https://xeno-canto.org/), a community-curated archive of wildlife sounds.

## License

Released for academic and educational use. The adapted portions retain their original authors' licenses; see linked sources above.
