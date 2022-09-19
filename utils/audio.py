import os

import numpy as np
import librosa

import cv2

def openAudioFile(path, sample_rate=44100, as_mono=True, mean_substract=False):
    
    # Open file with librosa (uses ffmpeg or libav)
    sig, rate = librosa.load(path, sr=sample_rate, mono=as_mono)

    # Noise reduction?
    if mean_substract:
        sig -= sig.mean()

    return sig, rate

# 切割1秒
def splitSignal(sig, rate, seconds, overlap, minlen):
    # print(seconds)
    # print(overlap)
    # overlap = 0.25 minlen = 1.0 seconds = 1 length = 1.0
    # Split signal with overlap
    sig_splits = []
    # print(f"len(sig) = {len(sig)}")
    # print(f"sig[0] = {sig[0]}")
    # print(f"sig[1] = {sig[1]}")
    # print(f"sig[2] = {sig[2]}")
    for i in range(0, len(sig), int((seconds - overlap) * rate)):
        split = sig[i:i + int(seconds * rate)] # 1秒
        # print(f"i = {i}")
        # print(f"split = {split}")
        # print(f"len split = {len(split)}")

        # End of signal?
        if len(split) < int(minlen * rate):
            break
        
        # Signal chunk too short?
        if len(split) < int(rate * seconds):
            split = np.hstack((split, np.zeros((int(rate * seconds) - len(split),))))
        
        sig_splits.append(split)

    # print(f"sig_splits = {sig_splits}")
    return sig_splits

def melspec(sig, rate, shape=(128, 256), fmin=500, fmax=15000, normalize=True, preemphasis=0.95):

    # shape = (height, width) in pixels

    # Mel-Spec parameters
    SAMPLE_RATE = rate
    N_FFT = shape[0] * 8 # = window length #1024
    N_MELS = shape[0] #128 nmfcc
    HOP_LEN = len(sig) // (shape[1] - 1)  #172
    # print(f"len(sig) = {len(sig)}")
    # print(f"HOP_LEN = {HOP_LEN}")
    FMAX = fmax
    FMIN = fmin
    # print("N_FFT = " , N_FFT)
    # print("N_MELS = " , N_MELS)

    # Preemphasis as in python_speech_features by James Lyons
    if preemphasis:
        sig = np.append(sig[0], sig[1:] - preemphasis * sig[:-1])

    # Librosa mel-spectrum
    melspec = librosa.feature.melspectrogram(y=sig, sr=SAMPLE_RATE, hop_length=HOP_LEN, n_fft=N_FFT, n_mels=N_MELS, fmax=FMAX, fmin=FMIN, power=1.0)
    
    # Convert power spec to dB scale (compute dB relative to peak power)
    melspec = librosa.amplitude_to_db(melspec, ref=np.max, top_db=80)

    # Flip spectrum vertically (only for better visialization, low freq. at bottom)
    melspec = melspec[::-1, ...]

    # Trim to desired shape if too large
    melspec = melspec[:shape[0], :shape[1]]

    # Normalize values between 0 and 1
    if normalize:
        melspec -= melspec.min()
        if not melspec.max() == 0:
            melspec /= melspec.max()
        else:
            mlspec = np.clip(melspec, 0, 1)

    return melspec.astype('float32')

def stft(sig, rate, shape=(128, 256), fmin=500, fmax=15000, normalize=True):

    # shape = (height, width) in pixels

    # STFT-Spec parameters
    N_FFT = int((rate * shape[0] * 2) / abs(fmax - fmin)) + 1
    P_MIN = int(float(N_FFT / 2) / rate * fmin) + 1
    P_MAX = int(float(N_FFT / 2) / rate * fmax) + 1    
    HOP_LEN = len(sig) // (shape[1] - 1)

    # Librosa stft-spectrum
    spec = librosa.core.stft(sig, hop_length=HOP_LEN, n_fft=N_FFT, window='hamm')

    # Convert power spec to dB scale (compute dB relative to peak power)
    spec = librosa.amplitude_to_db(librosa.core.magphase(spec)[0], ref=np.max, top_db=80)

    # Trim to desired shape using cutoff frequencies
    spec = spec[P_MIN:P_MAX, :shape[1]]

    # Flip spectrum vertically (only for better visialization, low freq. at bottom)
    spec = spec[::-1, ...]    

    # Normalize values between 0 and 1
    if normalize:
        spec -= spec.min()
        if not spec.max() == 0:
            spec /= spec.max()
        else:
            spec = np.clip(spec, 0, 1)    
    
    return spec.astype('float32')

def get_spec(sig, rate, shape, spec_type='linear', **kwargs):

    if spec_type.lower()== 'melspec':
        return melspec(sig, rate, shape, **kwargs)
    else:
        return stft(sig, rate, shape, **kwargs)

def signal2noise(spec):

    # Get working copy
    spec = spec.copy()
    
    # print(f'spec = {spec}')

    # Calculate median for columns and rows
    col_median = np.median(spec, axis=0, keepdims=True)
    row_median = np.median(spec, axis=1, keepdims=True)

    # print(f'row_median {row_median}')
    
    
    # print(spec < row_median * 1.25)

    # Binary threshold
    spec[spec < row_median * 1.25] = 0.0
    spec[spec < col_median * 1.15] = 0.0
    spec[spec > 0] = 1.0

    # Median blur
    spec = cv2.medianBlur(spec, 3)

    # Morphology
    spec = cv2.morphologyEx(spec, cv2.MORPH_CLOSE, np.ones((3, 3), np.float32))

    # Sum of all values
    spec_sum = spec.sum()

    # Signal to noise ratio (higher is better)
    
    
    try:
        s2n = spec_sum / (spec.shape[0] * spec.shape[1] * spec.shape[2])
        # print(f'{spec.shape[0]}, {spec.shape[1]}, {spec.shape[2]}')
    except:
        s2n = spec_sum / (spec.shape[0] * spec.shape[1])
        # print(f'{spec.shape[0]}, {spec.shape[1]}')


    return s2n

def specsFromSignal(sig, rate, shape, seconds, overlap, minlen, **kwargs):
    # overlap = 0.0 minlen = 1.0 seconds = length = 1.0

    # Split signal in consecutive chunks with overlap
    sig_splits = splitSignal(sig, rate, seconds, overlap, minlen)
    # print(len(sig_splits))

    # Extract specs for every sig split
    for sig in sig_splits:
        # print(f"sig = {sig}")
        # print(f"len(sig) = {len(sig)}")

        # Get spec for signal chunk
        spec = get_spec(sig, rate, shape, **kwargs)

        yield spec

def specsFromFile(path, rate, seconds, overlap, minlen, shape, start=-1, end=-1, **kwargs):

    # Open file
    sig, rate = openAudioFile(path, rate)
            
    # print(f"sig = {sig}")
    # print(f"rate = {rate}")
    # print(f"len(sig) = {len(sig)}")

    # Trim signal?
    if start > -1 and end > -1:
        sig = sig[int(start * rate):int(end * rate)]
        minlen = 0

    # Yield all specs for file
    for spec in specsFromSignal(sig, rate, shape, seconds, overlap, minlen, **kwargs):
        yield spec
    
if __name__ == '__main__':

    
    for spec in specsFromFile('../example/Acadian Flycatcher.wav',
                              rate=48000,
                              seconds=1,
                              overlap=0,
                              minlen=1,
                              shape=(128, 256),
                              fmin=500,
                              fmax=22500,
                              spec_type='melspec'):

        # Calculate and show noise measure
        noise = signal2noise(spec)
        # print(noise)

        # Show spec and wait for enter key
        cv2.imshow('SPEC', spec)
        cv2.waitKey(-1)
