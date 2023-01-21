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
    return stats.skew(signal, nan_policy="omit")


def get_kurtosis(signal : ArrayLike) -> ArrayLike:
    """Return kurtosis of digital signal"""
    return stats.kurtosis(signal, nan_policy="omit")


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


def mel_freq_cepstrum_coef(signal : ArrayLike, n_fft, n_mfcc=20) -> ArrayLike:
    return librosa.feature.mfcc(y=signal, n_mfcc=n_mfcc, n_fft=n_fft)


def spectral_centroid(signal : ArrayLike, n_fft) -> ArrayLike:
    return librosa.feature.spectral_centroid(y=signal, n_fft=n_fft)


def spectral_bandwith(signal : ArrayLike, n_fft) -> ArrayLike:
    return librosa.feature.spectral_bandwidth(y=signal, n_fft=n_fft)

#
# Harmonic contrast, not melody progression
#
def spectral_contrast(signal : ArrayLike, n_fft) -> ArrayLike:
    return librosa.feature.spectral_contrast(y=signal, n_fft=n_fft)

#
# Not used for melodies
#
def spectral_flatness(signal : ArrayLike, n_fft) -> ArrayLike:
    return librosa.feature.spectral_flatness(y=signal, n_fft=n_fft)

#
# Not used for melodies
#
def spectral_rollof(signal : ArrayLike, n_fft) -> ArrayLike:
    return librosa.feature.spectral_rolloff(y=signal, n_fft=n_fft)


def chroma_energy(signal: ArrayLike, n_fft)->ArrayLike:
    return librosa.feature.chroma_cens(y=signal, n_fft = n_fft)


def chroma_stft(signal: ArrayLike, n_fft)->ArrayLike:
    return librosa.feature.chroma_stft(y=signal, n_fft = n_fft)


def melody_contour(signal):
    return librosa.effects.harmonic(y=signal)


def fundamental_frequency(signal : ArrayLike) -> ArrayLike:
    f0 = librosa.yin(signal, fmin=20, fmax=11025)
    f0[f0==11025] = 0
    return f0

#
# Not good for melodies
#
def root_mean_square(signal: ArrayLike) -> ArrayLike:
    return librosa.feature.rms(y=signal)

#
# Not good for melodies
#
def zero_crossing_rate(signal: ArrayLike) -> ArrayLike:
    return librosa.feature.zero_crossing_rate(y=signal)


def get_tempo(signal: ArrayLike) -> ArrayLike:
    return librosa.beat.tempo(y=signal)


def get_feature_vector(signal: ArrayLike, n_fft: int) -> ArrayLike:
    if len(signal.shape)>1: #must be single channel
        if signal.shape[1]==1:
            signal = signal.flatten()
        else:
            signal = signal.mean(axis=1)

    if np.issubdtype(signal.dtype, np.integer): # or any type of int, in reality
        signal = librosa.util.buf_to_float(signal)
    
    mfcc = mel_freq_cepstrum_coef(signal, n_fft)
    centroid = spectral_centroid(signal, n_fft)
    bdwth = spectral_bandwith(signal, n_fft)
    harmony = chroma_stft(signal, n_fft)
    contour = melody_contour(signal)
    contour_intervals = chroma_stft(contour, n_fft)
    f0 = fundamental_frequency(signal)


    #contrast = spectral_contrast(signal, n_fft)
    #flatness = spectral_flatness(signal, n_fft)
    #rollof = spectral_rollof(signal, n_fft)
    #rms = root_mean_square(signal)
    #zcr = zero_crossing_rate(signal)
    #tempo = get_tempo(signal)

    mfcc = extract_statistics(mfcc)
    centroid = extract_statistics(centroid)
    bdwth = extract_statistics(bdwth)
    harmony = extract_statistics(harmony)
    contour_intervals = extract_statistics(contour_intervals)
    f0 = extract_statistics(f0, axis=0)

    #contrast = extract_statistics(contrast)
    #flatness = extract_statistics(flatness)
    #rollof = extract_statistics(rollof)
    #rms = extract_statistics(rms)
    #zcr = extract_statistics(zcr)

    return np.concatenate((
        mfcc, 
        #centroid,
        #bdwth,
        #harmony,
        contour_intervals,
        f0, 
        # contrast,
        # flatness,
        # rollof,
        # rms,
        # zcr,
        # tempo
    ))




