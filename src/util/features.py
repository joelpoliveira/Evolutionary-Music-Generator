import librosa
import numpy as np
import scipy.stats as stats
from numpy.typing import ArrayLike


def get_mean(signal : ArrayLike) -> float:
    """Return mean of digital signal"""
    return signal.mean()


def get_std(signal : ArrayLike) -> float:
    """Return standard deviation of digital signal"""
    return signal.std()


def get_median(signal: ArrayLike) -> float:
    """Return median of digital signal"""
    return np.median(signal)


def get_max(signal : ArrayLike) -> float:
    """Return max. value of digital signal"""
    return signal.max()


def get_min(signal : ArrayLike) -> float:
    """Return min. value of digital signal"""
    return signal.min()

def get_statistics(signal: ArrayLike) -> tuple:
    return (
        get_mean(signal),
        get_std(signal),
        get_median(signal),
        get_max(signal),
        get_min(signal),
    )


def extract_statistics(signal: ArrayLike, axis=1) -> ArrayLike:
    return np.apply_along_axis(get_statistics, axis, signal).flatten()


def mel_freq_cepstrum_coef(signal : ArrayLike, n_fft, sr, n_mfcc=10) -> ArrayLike:
    return librosa.feature.mfcc(y=signal, n_mfcc=n_mfcc, n_fft=n_fft, sr=sr)


def spectral_bandwith(signal : ArrayLike, n_fft, sr) -> ArrayLike:
    return librosa.feature.spectral_bandwidth(y=signal, n_fft=n_fft, sr=sr)


def spectral_contrast(signal : ArrayLike, n_fft, sr) -> ArrayLike:
    return librosa.feature.spectral_contrast(y=signal, n_fft=n_fft, sr=sr)


def chroma_energy(signal: ArrayLike, n_fft, sr)->ArrayLike:
    return librosa.feature.chroma_cens(y=signal, n_fft = n_fft, sr=sr)


def chroma_stft(signal: ArrayLike, n_fft, sr)->ArrayLike:
    return librosa.feature.chroma_stft(y=signal, n_fft = n_fft, sr=sr)


def fundamental_frequency(signal : ArrayLike, sr) -> ArrayLike:
    f0 = librosa.yin(signal, fmin=20, fmax=11025, sr=sr)
    f0[f0==11025] = 0
    return f0


def root_mean_square(signal: ArrayLike) -> ArrayLike:
    return librosa.feature.rms(y=signal)


def get_tempo(signal: ArrayLike, sr) -> ArrayLike:
    return librosa.beat.tempo(y=signal, sr=sr)


def get_feature_vector(signal: ArrayLike, n_fft: int, SR:int) -> ArrayLike:
    if len(signal.shape)>1: #must be single channel
        if signal.shape[1]==1:
            signal = signal.flatten()
        else:
            signal = signal.mean(axis=1)

    if np.issubdtype(signal.dtype, np.integer): # or any type of int, in reality
        signal = librosa.util.buf_to_float(signal)
    
    mfcc = mel_freq_cepstrum_coef(signal, n_fft, SR, 5)
    contrast = spectral_contrast(signal, n_fft, SR)
    bdwth = spectral_bandwith(signal, n_fft, SR)
    harmony = chroma_stft(signal, n_fft, SR) # pitches of western music
    f0 = fundamental_frequency(signal, SR)
    rms = root_mean_square(signal)
    tempo = get_tempo(signal, SR)

    

    mfcc = extract_statistics(mfcc)
    contrast = extract_statistics(contrast)
    bdwth = extract_statistics(bdwth)
    harmony = extract_statistics(harmony)
    f0 = extract_statistics(f0, axis=0)
    rms = extract_statistics(rms)

    return np.concatenate((
        mfcc, 
        contrast,
        bdwth,
        harmony,
        f0, 
        rms,
        tempo
    ))




