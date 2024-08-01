import numpy as np
import cupy as cp
from tqdm.auto import tqdm
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
    """
    A sandbox function, adds some white noise to the track.

    Parameters
    ----------
    np_samples : np.array of int, 1D
        The wav track.

    Returns
    -------
    noisy : np.array of int, 1D
        The noisy wav track.

    """
    noise = np.random.normal(0,1,len(np_samples))
    dynamic_range = np.max(np.abs(np_samples))
    noisy = np_samples + noise/20*dynamic_range
    return noisy


def show_spectra(np_samples,sampling_rate,flim=600):
    """
    A simple spectrogram of the track

    Parameters
    ----------
    np_samples : np.array of int, 1D
        The wav track.
    sampling_rate : float or int
        The sampling rate of the track.

    Returns
    -------
    None.

    """
    f, t, Sxx = signal.spectrogram(np_samples,sampling_rate,
                                   window='tukey',nperseg=64000)
    f_needed = int(flim/(sampling_rate/2)*len(f)+1)
    plt.pcolormesh(t, f[:f_needed],np.log10(Sxx[:f_needed,:]),
                   shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.ylim(0,flim)
    plt.show()

    S = np.sum(Sxx,axis=1)[:len(f)//2]
    f_drawn = np.random.choice(f[:len(f)//2],size=1000000,
                               replace=True,p=S/np.sum(S))
    plt.hist(f_drawn,bins=np.linspace(100,300,150))
    plt.xlabel("Feq. [Hz]")
    plt.show()


def approximate_with_frequencies_fft(np_samples,
                                     sampling_rate,
                                     frequency_width=20,
                                     list_of_frequencies=[116.54,
                                                          138.59,
                                                          155.56,
                                                          207.65,
                                                          233.08]):
    """
    Only keeps the frequencies close to the defined frequencies, 
    using cupy fft.

    Parameters
    ----------
    np_samples : np.array of int, 1D
        The wav track.
    sampling_rate : float or int
        The sampling rate of the track in Hz.
    frequency_width : floar ot int, optional
        Width of the filter, HWHM in Hz. The default is 20.
    list_of_frequencies : list of float, optional
        The frequencies to be allowed. The default is [116.54, 138.59, 155.56,
                                                       207.65, 233.08]                                                          138.59,                                                          155.56,                                                          207.65,                                                          233.08].

    Returns
    -------
    np_filtered : np.array of int, 1D
        The filtered wav track.

    """

    cp_samples = cp.array(np_samples)
    cp_fourier = cp.fft.fft(cp_samples)

    N = cp.size(cp_samples)
    n = cp.arange(N)
    T = N/sampling_rate
    freq = n/T

    mask = cp.zeros(cp.size(cp_fourier))
    for f in list_of_frequencies:
        mask += cp.exp(-((freq-f)/frequency_width)**2)
    np_filtered = cp.asnumpy(cp.abs(cp.fft.ifft(cp_fourier*mask)))

    return np_filtered


def approximate_with_frequencies(np_samples,
                                 sampling_rate,
                                 segment_length=100,
                                 frequency_half_width=20,
                                 list_of_frequencies=np.array([116.54,
                                                               138.59,
                                                               155.56,
                                                               207.65,
                                                               233.08])):
    """
    Similar to the fft alternative approximate_with_frequencies_fft, but 
    this one evaluates the presence of individual tones and then approximates
    the track with pure tones, segment-wise and in temporal domain, using 
    numpy.

    Parameters
    ----------
    np_samples : np.array of int, 1D
        The wav track.
    sampling_rate : float or int
        The sampling rate of the track in Hz.
    segment_length : float or int
        The length of one segment in ms.
    frequency_width : floar ot int, optional
        Width of each of the band-pass filters, HWHM in Hz. The default is 20.
    list_of_frequencies : list of float, optional
        The frequencies to be allowed. The default is [116.54, 138.59, 155.56,
                                                       207.65, 233.08]                                                          138.59,                                                          155.56,                                                          207.65,                                                          233.08].

    Returns
    -------
    np_filtered : np.array of int, 1D
        The filtered wav track.

    """

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


def approximate_with_chords(np_samples,
                            sampling_rate,
                            frequency_half_width=20,
                            list_of_chords=[[116.55,174.62,233.08,
                                             277.19,349.23,466.17],
                                            [138.60,174.62,207.65,
                                             277.19,349.23,466.17],
                                            [155.57,233.08,311.13,
                                             349.23,466.17],
                                            [185.00,233.08,277.19,
                                             369.99,466.17]]):
    """
    Approximates a track with given chords, 
    modulated by their presence in the track.

    Parameters
    ----------
    np_samples : np.array of int, 1D
        The wav track.
    sampling_rate : float or int
        The sampling rate of the track, Hz.
    frequency_width : float ot int, optional
        Width of the filter, HWHM in Hz. The default is 3.
    list_of_chords : list of list of float, optional
        The list of chords, each element in turn is a list of tones. 
        The default is [A#2mi, C#3, D#3sus2, F#3].

    Returns
    -------
    track : np.array of int, 1D
        The approximated wav track.

    """

    length_ms = len(np_samples)/sampling_rate*1000
    # generate chords
    numer_of_tones = len([tone for chord in list_of_chords for tone in chord])
    print("Generating synthetic tones:")
    synthetic_chords = []
    with tqdm(total=numer_of_tones) as pbar:
        for list_of_frequencies in list_of_chords:
            chord = np.zeros(len(np_samples))
            a = np.linspace(0,length_ms,len(np_samples))
            for freq in list_of_frequencies:
                tone = np.sin(a*freq/1000*2*np.pi)
                chord += tone
                pbar.update(1)
            synthetic_chords.append(chord)

    # evaluate the presence of chords
    track = np.zeros(len(np_samples))
    print("Evaluating chords' presence and stiching the track:")
    for c,chord in enumerate(tqdm(list_of_chords)):
        approximated = approximate_with_frequencies_fft(np_samples,
                                                        sampling_rate,
                                                        frequency_half_width,
                                                        chord)
        presence = np.abs(approximated)
        chord_modulated = synthetic_chords[c][:]*presence
        track += chord_modulated

    return track



#%%
if __name__ == "__main__":
    np_samples = load_mp3_size_limit(audio_file)
    sampling_rate=get_sampling(audio_file)
    approximated_with_chords = approximate_with_chords(np_samples,
                                                       sampling_rate)
    save_wav(approximated_with_chords,
             sampling_rate,
             folder=audio_save_folder,
             name="approximated_chords.wav")
    show_spectra(approximated_with_chords,sampling_rate)


