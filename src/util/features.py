import librosa
import numpy as np
import scipy.stats as stats
from scipy.spatial.distance import cosine

def get_mean(signal : np.array) -> float:
    return signal.mean()

def get_std(signal : np.array) -> float:
    return signal.std()

def get_skewness(signal : np.array) -> np.array:
    return stats.skew(signal)

def get_kurtosis(signal : np.array) -> np.array:
    return stats.kurtosis(signal)

def get_median(signal: np.array) -> float:
    return np.median(signal)

def get_max(signal : np.array) -> float:
    return signal.max()

def get_min(signal : np.array) -> float:
    return signal.min()

def get_statistics(signal: np.array) -> tuple:
    return (
        get_mean(signal),
        get_std(signal),
        get_median(signal),
        get_max(signal),
        get_min(signal),
        get_skewness(signal),
        get_kurtosis(signal)
    )

def extract_statistics(signal: np.array, axis=1):
    return np.apply_along_axis(get_statistics, axis, signal).flatten()

def mel_freq_cepstrum_coef(signal : np.array):
    return librosa.feature.mfcc(y=signal, n_mfcc=13)

def spectral_centroid(signal : np.array):
    return librosa.feature.spectral_centroid(y=signal)

def spectral_bandwith(signal : np.array):
    return librosa.feature.spectral_bandwidth(y=signal)

def spectral_contrast(signal : np.array):
    return librosa.feature.spectral_contrast(y=signal)

def spectral_flatness(signal : np.array):
    return librosa.feature.spectral_flatness(y=signal)

def spectral_rollof(signal : np.array):
    return librosa.feature.spectral_rolloff(y=signal)

def fundamental_frequency(signal : np.array):
    f0 = librosa.yin(signal, fmin=20, fmax=11025)
    f0[f0==11025] = 0
    return f0

def root_mean_square(signal: np.array):
    return librosa.feature.rms(y=signal)

def zero_crossing_rate(signal: np.array):
    return librosa.feature.zero_crossing_rate(y=signal)

def get_tempo(signal: np.array):
    return librosa.beat.tempo(y=signal)

def get_feature_vector(signal: np.array) -> np.array:
    if signal.dtype==np.int16: # or any type of int, in reality
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




