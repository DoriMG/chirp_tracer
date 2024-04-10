""" This file contains all functions needed to perform the tracing

It expects all txt files are in a folder with accompanying wav files with the same name"""

import numpy as np
import matplotlib.pyplot as plt
import os
import soundfile as sf
import scipy
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from scipy import ndimage
import cv2
from sklearn.preprocessing import MinMaxScaler
import pandas as pd


def find_all_text_files(folders, check_sub_folders=1):
    """Find all files with a .txt extension in all folders in folder list (and subfolders if check_sub_folders = 1 (default))"""
    all_files = []
    for f, folder in enumerate(folders):
        for subfolder in os.listdir(folder):

            if os.path.isdir(os.path.join(folder, subfolder)):
                if check_sub_folders:
                    folders.append(os.path.join(folder, subfolder))                
            elif subfolder.endswith(".txt"):
                all_files.append(os.path.join(folder,subfolder))
    return all_files

def load_calls(file, filter_calls, call_type, delimiter='\t'):
    """Extract calls from txt outputed by Audacity.
    
    If you only want a certain type of call, set filter_calls to 1, and input the name of the call_type as it is named in the txt file (e.g. 's').

    """
    # read calls into pandas dataframe
    calls = pd.read_csv(file, delimiter='\t', names = ['start_time', 'end_time', 'call_type'])
    calls = calls.fillna('nan')

    # Put call type into list if not already
    if type(call_type) != list:
        call_type = [call_type]
    
    # Clean up call type
    calls['call_type'] = calls['call_type'].apply(lambda x: x.lower().strip() if isinstance(x, str) else x)
    clean_call_type = [s.lower().strip() for s in call_type]

    # filter the pandas dataframe
    if filter_calls:
        mask = calls['call_type'].isin(clean_call_type)
        filtered_calls = calls[mask]
    else:
        filtered_calls = calls
    
    return filtered_calls

def load_wav_file(file):
    """Finds the wav file with the same name as the .txt file"""

    dataset = os.path.splitext(file)[0]
    folder_name, fname = os.path.split(dataset)
    data, samplerate = sf.read(os.path.join(folder_name, fname+'.wav'))
    if data.ndim>1:
        data = data[:,0]
    return data, samplerate

def extract_spectrograms(calls, wav_data, samplerate, lowpass = 2000, highpass = 30000, NFFT=512, noverlap=256):
    """Extract spectrogram and wav file for each call"""
    spectrograms = []
    call_wavs = []
    for index, call in calls.iterrows():
        t_start = int(call['start_time']*samplerate)
        t_end = int(call['end_time']*samplerate)
        call_wav = wav_data[t_start:t_end]

        [spec, freq, t, im] = plt.specgram(call_wav,Fs=samplerate, NFFT=NFFT, noverlap=noverlap)
        inc = np.logical_and(freq > lowpass,freq < highpass)
        plt.close()

        spectrograms.append(np.log(spec[inc,:]))
        call_wavs.append(call_wav)

    calls['spectrogram'] = spectrograms
    calls['wav'] = call_wavs
    return calls, freq[inc]


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx

def get_harmonics_matrix(freq, max_fun):
    fun_freq_range = freq<max_fun
    fun_freq = freq[fun_freq_range]

    # For each fun frequency, what is the harmonic frequency?
    # then, put that in an vector with 2 peaks

    fun_fits = np.zeros((np.sum(fun_freq_range), len(freq)))
    fun_fits_gauss = np.zeros((np.sum(fun_freq_range), len(freq)))
    for i,f in enumerate(fun_freq):
        f_harm, idx_harm = find_nearest(freq, value=f*2)
        fun_fits[i, i] = 10
        fun_fits[i, idx_harm] = 1
        fun_fits_gauss[i,:] = gaussian_filter1d(fun_fits[i,:], 3)
    
    return fun_fits_gauss

def denoise_call(spec):
    """Denoise the spectrogram image"""
    # do this to change to uint8
    dfmax, dfmin = spec.max(), spec.min()
    call = (spec - dfmin)/(dfmax - dfmin)
    call = call*255

    denoised_call = cv2.fastNlMeansDenoising(call.astype(np.uint8), None, 30, 7, 9)

    return denoised_call

def convolve_call(denoised_call, fun_fits_gauss):
    """Convolve the image with harmonics matrix to obtain harmonics corrected power"""
    convolved = np.empty((denoised_call.shape[1],fun_fits_gauss.shape[0]))
    for i, col in enumerate(denoised_call.T):
        fit_harms = np.matmul(fun_fits_gauss,col)
        convolved[i,:] = fit_harms
    
    return convolved


def smart_trace(convolved_call, fun_freq, mod=0.5,zscore_thresh=0):
    def_list  = list(range(convolved_call.shape[1])) # list of all frequencies
    new_trace = np.empty(fun_freq.shape) # this is where new trace will be saved
    new_trace[0] = fun_freq[0]

    for i, f in enumerate(new_trace[:-1]):
        # get the next timepoint from call
        next_freq = convolved_call[i+1,:]

        # calculate the parabola
        dist_to_f = (def_list-f)**2
        dist_to_f = -1*dist_to_f

        # check that the current z-score is still high enough (i.e. I'm confident about this point)
        current_zscore = stats.zscore(convolved_call[i,:])[int(f)]
        if current_zscore>zscore_thresh:
            # scale the scaler to my current z-score
            scaler = MinMaxScaler(feature_range=(1, 1+(mod*current_zscore)))
            x_scaled = scaler.fit_transform(dist_to_f.reshape((-1, 1)))
            
            # Get corrected frequency
            next_freq_corr = np.multiply(next_freq.reshape((-1,1)),x_scaled)
        else:
            next_freq_corr = next_freq
        new_trace[i+1] = np.argmax(next_freq_corr) # add to trace
    return new_trace

def trace_call(calls, fun_fits_gauss, mod = 0.5):
    denoised_calls = []
    convolved_calls = []
    old_traces = []
    new_traces = []
    for index, call in calls.iterrows():
        denoised_call = denoise_call(call['spectrogram'])
        convolved = convolve_call(denoised_call, fun_fits_gauss)
        old_trace = np.argmax(convolved, axis=1)
        new_trace = smart_trace(convolved, old_trace,mod)

        denoised_calls.append(denoised_call)
        convolved_calls.append(convolved)
        old_traces.append(old_trace)
        new_traces.append(new_trace)
    
    calls['denoised'] = denoised_calls
    calls['convolved'] = convolved_calls
    calls['raw_trace'] = old_traces
    calls['final_trace'] = new_traces

    return calls
