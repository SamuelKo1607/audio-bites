import numpy as np
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 600
import matplotlib.pyplot as plt
from numba import jit
from paths import audio_file
from paths import audio_save_folder
from scipy import signal
from load import load_mp3_size_limit
from load import load_mp3_to_numpy
from load import save_wav
from load import get_sampling


def add_white_noise(np_samples):
    noise = np.random.normal(0,1,len(np_samples))
    dynamic_range = np.max(np.abs(np_samples))
    noisy = np_samples + noise/20*dynamic_range
    return noisy


def show_spectra(np_samples):
    rate = get_sampling(audio_file)
    f, t, Sxx = signal.spectrogram(np_samples,rate,
                                   window='tukey',nperseg=64000)
    plt.pcolormesh(t, f, Sxx, shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.ylim(0,1000)
    plt.show()

    S = np.sum(Sxx,axis=1)[:len(f)//2]
    f_drawn = np.random.choice(f[:len(f)//2],size=1000000,
                               replace=True,p=S/np.sum(S))
    plt.hist(f_drawn,bins=np.linspace(100,300,150))
    plt.show()


def approximate_with_frequencies_fft(np_samples,
                                     sampling_rate, #Hz
                                     frequency_width=10, #Hz
                                     list_of_frequencies=[116.54, #Hz
                                                          138.59,
                                                          155.56,
                                                          207.65,
                                                          233.08]):
    pass #TBD calculate fft, then change the fourier image, then inverse


def approximate_with_frequencies(np_samples,
                                 sampling_rate, #Hz
                                 segment_length=100, #ms
                                 frequency_half_width=20, #Hz
                                 list_of_frequencies=np.array([116.54, #Hz
                                                               138.59,
                                                               155.56,
                                                               207.65,
                                                               233.08])):
    length_ms = len(np_samples)/sampling_rate*1000
    number_of_segments = int(length_ms/segment_length)
    segment_length_samples = int(sampling_rate/1000*segment_length)

    # generate tones
    synthetic_tracks = []
    for freq in list_of_frequencies:
        a = np.linspace(0,length_ms,len(np_samples))
        tone = np.sin(a*freq/1000*2*np.pi)
        synthetic_tracks.append(tone)

    # calculate the coefficients
    smoothing_kernel = 2/(1+np.exp(-50*np.linspace(0,1,
                                                   segment_length_samples)))-1
    smoothing_kernel *= np.flip(smoothing_kernel)
    power_coefficients = np.zeros((number_of_segments,
                                   len(list_of_frequencies)))
    for segment_number in range(number_of_segments):
        segment = np_samples[segment_number*segment_length_samples:
                             (segment_number+1)*segment_length_samples]
        segment = segment * smoothing_kernel
        for i,freq in enumerate(list_of_frequencies):
            b, a = signal.butter(2,
                                 (freq-frequency_half_width,
                                  freq+frequency_half_width),
                                 'bandpass',
                                 fs=sampling_rate)
            filtered = signal.filtfilt(b, a, segment)
            power = np.mean(filtered**2)
            power_coefficients[segment_number,i] = power

    # produce the audio signal based on the coefficients
    x_full = np.arange(len(np_samples))
    x_reduced = np.linspace(0,x_full[-1],num=number_of_segments)
    np_samples_output = np.zeros(len(np_samples))
    for i,freq in enumerate(list_of_frequencies):
        oversampled_coefficients = np.interp(x_full,
                                             x_reduced,
                                             power_coefficients[:,i])
        modulated_tone = synthetic_tracks[i] * oversampled_coefficients
        np_samples_output = np_samples_output + modulated_tone

    # lo-pass to get rid of the mess
    b, a = signal.butter(4,
                         list_of_frequencies[-1]*1.2,
                         'lowpass',
                         fs=sampling_rate)
    lopassed = signal.filtfilt(b, a, np_samples_output)

    return lopassed




#%%
if __name__ == "__main__":

    np_samples = load_mp3_size_limit(audio_file)
    sampling_rate=get_sampling(audio_file)

    approximated = approximate_with_frequencies(np_samples,sampling_rate)
    save_wav(approximated,
             sampling_rate,
             folder=audio_save_folder,
             name="approximated.wav")

    show_spectra(np_samples)
    show_spectra(approximated)

    noisy = add_white_noise(np_samples)
    save_wav(noisy,
             sampling_rate,
             folder=audio_save_folder,
             name="noisy.wav")





