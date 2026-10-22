# 🐦 BirdSoundRecognize

> Deep learning pipeline for bird species identification from audio recordings — a master's thesis project using mel-spectrograms and a custom ResNet architecture.

<p align="center">
  <img src="image/model%20flow%20chart.png" alt="System flow chart" width="720">
</p>

**Companion mobile app:** [bird-sound-recognize-app](https://github.com/Bobo100/bird-sound-recognize-app) — runs the trained TFLite model on Android for real-time on-device inference.

---

## What it does

1. **Converts** bird recordings (e.g. from [xeno-canto](https://xeno-canto.org/)) from mp3 to wav.
2. **Extracts** mel-spectrograms as 256×128 grayscale images.
3. **Trains** a custom ResNet-style CNN (Keras / TensorFlow) to classify species from spectrograms.
4. **Exports** the model as TFLite for the companion Android app.

## Tech stack

| Layer | Tool |
|---|---|
| Framework | TensorFlow / Keras 3 |
| Architecture | Custom pre-activation ResNet (4 stacks × 2 ResBlocks, filters 32–256) |
| Audio features | Mel-spectrogram (`librosa`), 44.1 kHz, 1 s windows, 500–15000 Hz |
| Optimizer | Nadam, `ReduceLROnPlateau` on validation loss |
| Imbalance handling | Log-smoothed class weights |

The original 2022 runs used Python 3.7 on Windows 10 (i7-6700, GTX 1070). The code now runs on any OS with Python ≥ 3.10.

## Repository structure

```
.
├── config.py           # All paths and hyperparameters
├── convert_mp3.py      # mp3 → wav conversion
├── extract_specs.py    # wav → mel-spectrogram PNGs
├── train.py            # Training loop, checkpoints, TFLite export
├── model.py            # ResNet architecture
├── utils/
│   ├── audio.py        # Audio loading, chunking, spectrograms, signal-to-noise
│   ├── training.py     # Class weights, checkpoint resume, TFLite export
│   └── log.py
├── data/
│   ├── audio/          # Input wav files, one folder per species (3 sample species included)
│   ├── specs/          # Generated spectrograms, one folder per species
│   └── noise/          # Rejected low-signal spectrograms
├── models/
│   └── resnet_0919_test/   # Reference TFLite model + labels
├── docs/plan/          # Records of why the project was changed
└── image/              # README assets
```

## Quick start

```bash
pip install -r requirements.txt
```

### 1. Prepare the dataset

Organize your audio files by species (any number of classes):

```
data/audio/
├── species1/
│   ├── species1_1.wav
│   └── species1_2.wav
└── species2/
    └── species2_1.wav
```

If your files are mp3 (needs `ffmpeg`), convert them first:

```bash
python convert_mp3.py path/to/mp3_folder        # writes to data/audio/
```

### 2. Extract spectrograms

```bash
python extract_specs.py
```

### 3. Train

```bash
python train.py
```

Everything is written to `models/<RUN_NAME>/`:

| File | Content |
|---|---|
| `checkpoints/weights.NN.keras` | Full model after every epoch |
| `training_log.csv` | Loss / accuracy per epoch |
| `resnet_model.tflite`, `resnet_labels.txt` | Latest model for the Android app |

Running `train.py` again resumes from the newest checkpoint. Change `RUN_NAME` in `config.py` to start a new run.

## How it works

1. **Audio → image**: each clip is split into 1.0 s windows (0.25 s overlap) and converted to a mel-spectrogram in the 500–15000 Hz band, normalized to a 256×128 image.
2. **Noise vs signal**: chunks below a signal threshold are rejected; 10 % of them are kept in `data/noise/` (intended for noise augmentation, which the current training does not use).
3. **Model**: a pre-activation ResNet. Filter counts, kernel size and stack depth are set in `config.py`.
4. **Input format**: `128 × 256 × 1` grayscale with pixel values in **[0, 255]** (no rescaling) — the Android app feeds the model in the same range.
5. **Output**: softmax probability per species; label order is in `*_labels.txt`.

## Status

This project was part of a master's thesis (2022) and is **not actively maintained**. In 2026 the code was restructured and a class-weight bug was fixed; the reasoning is recorded in [`docs/plan/`](docs/plan/). The included reference model was trained before that fix.

## Acknowledgements & references

- **Spectrogram extraction & ResNet design** — adapted from [Stefan Kahl's BirdCLEF-Baseline](https://github.com/kahst/BirdCLEF-Baseline) (originally Lasagne; rewritten here for TensorFlow / Keras).
- **Dataset** — [xeno-canto](https://xeno-canto.org/), a community-curated archive of wildlife sounds.

## License

Released for academic and educational use. The adapted portions retain their original authors' licenses; see linked sources above.
