# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 14:27:11 2024

@author: juryl
"""

from scipy import signal
import numpy as np
from scipy.signal import find_peaks


# de-noising oscillogram
def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def refine_lims(sig,samplerate,hpass=2000,dist_peaks=30,threshold=0.00045):
    
    # 'sig' should be an array with the soft chirp oscillogram
    # 'samplerate' (in Hz)
    # 'hpass' is the cut frequency of the high pass filter
    # 'dist_peaks' (in samples) is the refractory period to look for peaks
    # 'threshold' (in V) is the amplitude of the signal to be considered a call
    
    # filter the oscillogram
    sig = butter_highpass_filter(sig,hpass,samplerate)
    # de-mean the signal
    sig=sig-np.mean(sig)
    # calculate the envelope as interpolation of peaks
    peaks, _ = find_peaks(sig, height=0, distance=dist_peaks)
    time=np.linspace(0,len(sig)/samplerate,len(sig))
    env=np.interp(time,time[peaks], sig[peaks])
    # find first time at which envelope surpass the threshold
    logic=env>threshold
    init=np.argmax(logic)  # in samples
    # find last time at which envelope goes below the threshold
    reverse=logic[::-1]
    end=np.argmax(reverse)  
    # in case it never goes below the threshold
    if end==0:
        end=-1  # end should be the last sample

    # cut the call using the new limits
    cut_call=sig[init:-end]
    
    return cut_call

    
