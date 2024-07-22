import numpy as np
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 600
import matplotlib.pyplot as plt
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


def decompose_into_frequencies_fft(np_samples,
                                   sampling_rate, #Hz
                                   frequency_width=10, #Hz
                                   list_of_frequencies=[116.54, #Hz
                                                        138.59,
                                                        155.56,
                                                        207.65,
                                                        233.08]):
    pass #TBD calculate fft, then change the fourier impage, then inverse


def decompose_into_frequencies(np_samples,
                               sampling_rate, #Hz
                               segment_length=100, #ms
                               frequency_half_width=10, #Hz
                               list_of_frequencies=[116.54, #Hz
                                                    138.59,
                                                    155.56,
                                                    207.65,
                                                    233.08]):
    length_ms = len(np_samples)/sampling_rate*1000

    # generate tones
    synthetic_tracks = []
    for freq in list_of_frequencies:
        a = np.linspace(0,length_ms,len(np_samples))
        tone = np.sin(a*freq*2*np.pi)
        synthetic_tracks.append(tone)

    # calculate the coefficients
    smoothing_kernel = 2/(1+np.exp(-50*np.linspace(0,1,segment_length)))-1
    smoothing_kernel *= np.flip(smoothing_kernel)
    number_of_segments = int(len(np_samples)/segment_length)
    for segment_number in range(number_of_segments):
        segment = np_samples[segment_number*segment_length:
                             (segment_number+1)*segment_length]
        segment = segment * smoothing_kernel
        for freq in list_of_frequencies:
            b, a = signal.butter(3,
                                 (freq-frequency_half_width,
                                  freq+frequency_half_width),
                                 'bandpass',
                                 fs=sampling_rate)
            filtered = signal.filtfilt(b, a, segment)
            plt.plot(filtered)
            # tbd calculate remaining power




#%%
if __name__ == "__main__":
    np_samples = load_mp3_size_limit(audio_file)
    sampling_rate=get_sampling(audio_file)

    show_spectra(np_samples)

    noisy = add_white_noise(np_samples)
    save_wav(noisy,
             sampling_rate,
             folder=audio_save_folder,
             name="noisy.wav")





