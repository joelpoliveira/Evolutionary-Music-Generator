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

def get_skewness(signal : ArrayLike) -> ArrayLike:
    """Return skewness of digital signal"""
    return stats.skew(signal)

def get_kurtosis(signal : ArrayLike) -> ArrayLike:
    """Return kurtosis of digital signal"""
    return stats.kurtosis(signal)

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
        get_skewness(signal),
        get_kurtosis(signal)
    )

def extract_statistics(signal: ArrayLike, axis=1) -> ArrayLike:
    return np.apply_along_axis(get_statistics, axis, signal).flatten()

def mel_freq_cepstrum_coef(signal : ArrayLike, n_mfcc=13) -> ArrayLike:
    return librosa.feature.mfcc(y=signal, n_mfcc=n_mfcc)

def spectral_centroid(signal : ArrayLike) -> ArrayLike:
    return librosa.feature.spectral_centroid(y=signal)

def spectral_bandwith(signal : ArrayLike) -> ArrayLike:
    return librosa.feature.spectral_bandwidth(y=signal)

def spectral_contrast(signal : ArrayLike) -> ArrayLike:
    return librosa.feature.spectral_contrast(y=signal)

def spectral_flatness(signal : ArrayLike) -> ArrayLike:
    return librosa.feature.spectral_flatness(y=signal)

def spectral_rollof(signal : ArrayLike) -> ArrayLike:
    return librosa.feature.spectral_rolloff(y=signal)

def fundamental_frequency(signal : ArrayLike) -> ArrayLike:
    f0 = librosa.yin(signal, fmin=20, fmax=11025)
    f0[f0==11025] = 0
    return f0

def root_mean_square(signal: ArrayLike) -> ArrayLike:
    return librosa.feature.rms(y=signal)

def zero_crossing_rate(signal: ArrayLike) -> ArrayLike:
    return librosa.feature.zero_crossing_rate(y=signal)

def get_tempo(signal: ArrayLike) -> ArrayLike:
    return librosa.beat.tempo(y=signal)

def get_feature_vector(signal: ArrayLike) -> ArrayLike:
    if len(signal.shape)>1: #must be single channel
        if signal.shape[1]==1:
            signal = signal.flatten()
        else:
            signal = signal.mean(axis=1)

    if np.issubdtype(signal.dtype, np.integer): # or any type of int, in reality
        signal = librosa.util.buf_to_float(signal)
    
    mfcc = mel_freq_cepstrum_coef(signal)
    centroid = spectral_centroid(signal)
    bdwth = spectral_bandwith(signal)
    contrast = spectral_contrast(signal)
    flatness = spectral_flatness(signal)
    rollof = spectral_rollof(signal)
    f0 = fundamental_frequency(signal)
    rms = root_mean_square(signal)
    zcr = zero_crossing_rate(signal)
    tempo = get_tempo(signal)

    mfcc = extract_statistics(mfcc)
    centroid = extract_statistics(centroid)
    bdwth = extract_statistics(bdwth)
    contrast = extract_statistics(contrast)
    flatness = extract_statistics(flatness)
    rollof = extract_statistics(rollof)
    f0 = extract_statistics(f0, axis=0)
    rms = extract_statistics(rms)
    zcr = extract_statistics(zcr)

    return np.concatenate((
        mfcc, 
        centroid, 
        bdwth,
        contrast,
        flatness,
        rollof,
        f0,
        rms,
        zcr,
        tempo
    ))




