import os
import warnings

import math
import numpy as np
import soundfile as sf
import scipy

def acoustic_power(s):
    """ Acoustic power of after removing the silences.
    """
    w = 512  # Window size for silent detection
    o = 256  # Window step for silent detection

    # Window the input signal
    s = np.ascontiguousarray(s) # transform a data which is not continuous in the memory to be continuous
    sh = (s.size - w + 1, w)
    st = s.strides * 2
    S = np.lib.stride_tricks.as_strided(s, strides=st, shape=sh)[0::o] # s.shape [298, 512]

    window_power = np.mean(S ** 2, axis=-1) # ? the reason for this operation
    th = 0.01 * window_power.max()  # ? Threshold for silent detection
    return np.mean(window_power[np.nonzero(window_power > th)])


def add_noise(i, aud_file_path, save_path):
    SNR = 5 # change it!
    VAR = 10000 # change it!
    audio_data_, fs = sf.read(aud_file_path) #[1061282, 15]
    # For LOCATA DICIT ARRAY
    audio_data_ = np.concatenate((audio_data_[:,8:9], audio_data_[:,5:6]), axis=-1) # [ 1061282, 2]

    # chanege the sampling rate to 16000
    if fs > 16000:
        audio_data_ = scipy.signal.decimate(audio_data_, int(fs/16000), axis=0)
        new_fs = fs / int(fs/16000)
        if new_fs != 16000: warnings.warn('The actual fs is {}Hz'.format(new_fs))
        fs = new_fs
    elif fs < 16000:
        raise Exception('The sampling rate of the file ({}Hz) was lower than self.fs ({}Hz'.format(fs, 16000))
    
    # Gen Gaussian Noise with different variance
    noise_signal = np.random.normal(0, VAR, audio_data_.shape)
    mic_signals = audio_data_
    ac_pow = np.mean([acoustic_power(audio_data_[:,i]) for i in range(audio_data_.shape[1])])
    ac_pow_noise = np.mean([acoustic_power(noise_signal[:,i]) for i in range(noise_signal.shape[1])])
    noise_signal = np.sqrt(ac_pow/10**(SNR/10)) / np.sqrt(ac_pow_noise) * noise_signal # 同比例放大
    mic_signals += noise_signal[:, :] # shape: [76640, 2]
    
    # sf.write(os.path.join(save_path, str(i) +'origin' +'_.wav'), audio_data_, fs) 
    
    sf.write(os.path.join(save_path, str(i)+'_' +str(VAR) +'_.wav'), audio_data_, 16000)
    

def main():

    data_paths = []
    dataset_path = "/TSSL/locata_task_3_5/" # change it!
    save_path = "/TSSL/locata_task_3_5_noise/" # change it!
    data_names = os.listdir(dataset_path)
    for fname in data_names:
        front, ext = os.path.splitext(fname)
        if ext == ".wav":
            data_paths.append((os.path.join(dataset_path, fname)))
    data_paths.sort()
    audio_file_directory = data_paths
    total_number = len(audio_file_directory)
    
    for i in range(total_number):
        add_noise(i, audio_file_directory[i], save_path)    

if __name__ == "__main__":
    main()

