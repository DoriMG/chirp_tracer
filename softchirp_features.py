""" This file contains all functions needed to extract softchirp features from traced spectrograms"""

import numpy as np
import scipy
from librosa.core import piptrack
import librosa
import pandas as pd

def wiener_entropy(s_npy,  samplerate, n_fft = 512, low_pass = 2000, high_pass = 8000):
    """ Computes Wiener entropy of wav """

    freqs = librosa.fft_frequencies(sr=samplerate, n_fft=n_fft)
    inc = np.logical_and(freqs>low_pass, freqs<high_pass)
    
    S, phase = librosa.magphase(librosa.stft(s_npy,n_fft=n_fft, hop_length=256))
    ent = librosa.feature.spectral_flatness(S=S[inc,:])
    return np.var(ent), np.mean(ent), np.max(ent), ent


def spectral_contrast(s_npy, n_fft=2048):
    contrast = librosa.feature.spectral_contrast(y=s_npy, n_fft=n_fft)
    return np.mean(contrast), np.var(contrast), contrast

def goodness_of_pitch(s_npy, samplerate):
    mfcc = librosa.feature.mfcc(y=s_npy, sr=samplerate)
    mfcc_delta = librosa.feature.delta(mfcc)
    return np.max(mfcc_delta)

def compute_pitch(s_npy, samplerate,  win_length = 220):
    """ Computes pitch of wav sound """
    pitches, magnitudes = piptrack(y=s_npy, sr = samplerate, fmin= 2000, fmax=8000, win_length = win_length)
    bins = np.arange(pitches.shape[0])
    mean_pitch = np.mean((np.mean(pitches * magnitudes, 1)) * bins)
    return mean_pitch

def compute_amplitude(s_npy, win_length = 220):
    """ Compute average amplitued of wav """
    windows = [s_npy[i : i + win_length] for i in range(0, len(s_npy) + 1 - win_length, win_length)]
    amps = [np.max(w) - np.min(w) for w in windows]
    return np.mean(amps)

def compute_wiener_ent(s_npy, win_length = 220):
    """ Computes Wiener entropy of wav
     NOTE: This should not be used, I left it in for backwards compatability with AB's paper """
    ent = librosa.feature.spectral_flatness(y = s_npy, win_length = win_length)
    return np.sum(ent)

def compute_zero_crossing(s_npy,):
    """ Computes averate zero crossing ratio of wav """
    return np.mean(librosa.zero_crossings(s_npy, pad=False))

def compute_duration(call):
    """Computes duration of the trace (as image). Requires wav sound. """
    return call['end_time']-call['start_time']

def compute_height(t):
    """ Computes height of the trace (as image) in Hz """
    return (np.max(t[:,1]) - np.min(t[:,1])) 
    
def compute_asymmetry(t):
    """ Computes asymmetry between height of left and right ends of the trace (as image) in Hz """
    return t[0,1] - t[-1,1]

def parabola(x, a, h, k):
    return -(a*(x-h)**2+k)

def compute_slope(line):
    fit_params, pcov = scipy.optimize.curve_fit(parabola, line[:,0], line[:,1])
    slope =   fit_params[0]
    return slope

def compute_max_freq(line):
    return np.max(line[:,1])

def compute_all_features(calls, samplerate, freq, n_fft=256):
    """ Computes all the features above for list of traces and corresponding wavs """
    features = ['slope', 'frequency', 'pitch', 'amplitude', 
                                            'wiener_entropy', 'zero_crossings', 'duration', 
                                            'height', 'asymmetry', 'entropy_variance', 'mean_entropy', 'maximum_entropy', 'mean_contrast','var_contrast', 'goodness_of_pitch']
    fail_count = 0
    for newcol in features:
        calls[newcol]=np.nan

    for index, call in calls.iterrows():
        try:
            s = call['wav']
            t = np.asarray(list(zip(list(range(1,call['final_trace'].shape[0]+1)),call['final_trace']) ))
            
                
            calls.at[index,'pitch'] = compute_pitch(s, samplerate)
            calls.at[index,'amplitude']  = compute_amplitude(s)
            calls.at[index,'wiener_entropy']  = compute_wiener_ent(s)
            calls.at[index,'zero_crossings'] = compute_zero_crossing(s)
            calls.at[index,'slope'] = compute_slope(t)
            calls.at[index,'frequency'] = compute_max_freq(t)
            calls.at[index,'duration'] = compute_duration(call)
            calls.at[index,'height'] = compute_height(t)
            calls.at[index,'asymmetry'] = compute_asymmetry(t)
            
            ent_var, mean_ent, max_ent,  ent = wiener_entropy(s,samplerate, n_fft=512, low_pass = 2000, high_pass = 8000) 
            calls.at[index,'entropy_variance'] = ent_var
            calls.at[index,'mean_entropy'] = mean_ent
            calls.at[index,'maximum_entropy'] = max_ent

            mean_contrast, var_contrast, contrast= spectral_contrast(s, n_fft=256)    
            calls.at[index,'mean_contrast'] = mean_contrast
            calls.at[index,'var_contrast'] = var_contrast

            gop = goodness_of_pitch(s, samplerate)
            calls.at[index,'goodness_of_pitch'] = gop
        except:
            fail_count += 1
    
    print(str(fail_count) + ' calls failed')
        
    freq_p_step = np.nanmean(np.diff(freq))
    freq_adj = min(freq)

    calls['Frequency (Hz)'] = calls['frequency']*freq_p_step+freq_adj
    calls['Height (Hz)'] = calls['height'] * freq_p_step
    calls['Asymmetry (Hz)'] = calls['asymmetry'] * freq_p_step
    return calls