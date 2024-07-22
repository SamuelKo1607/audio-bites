import sys
import numpy as np
from pydub import AudioSegment
from scipy.io.wavfile import write
from paths import audio_file
from paths import audio_save_folder


def roundup(num,denom):
    return int(num / denom) + (num % denom > 0)


def get_sampling(file):
    samples_1s = load_mp3_to_numpy(file,sec_start=0,sec_end=1)
    return int(len(samples_1s))


def load_mp3_to_numpy(file,sec_start=0,sec_end=10):
    """
    Loads a portion of an MP3 into a numpy array.

    Parameters
    ----------
    file : str
        location of the MP3 file.
    sec_start : float, optional
        The starting second. The default is 0.
    sec_end : float or None, optional
        The end second. The default is 10. 
        If None, then the song until the end is returned.

    Returns
    -------
    np_samples : np.array of int, 1D
        The samples of the sound.

    """
    audio = AudioSegment.from_file(file, format="mp3")
    if sec_end is None:
        samples = audio[int(1000*sec_start):].get_array_of_samples()
    else:
        samples = audio[int(1000*sec_start):
                        int(1000*sec_end)].get_array_of_samples()
    np_samples = np.array(samples)
    return np_samples


def load_mp3_size_limit(file,limit_mb=256,chunk=0):
    """
    Loads an MP3 into a numpy array, chunking a possibly long file
    to manageable pieces.

    Parameters
    ----------
    file : str
        location of the MP3 file.
    limit_mb : float, optional
        The size limit of one chunk. The default is 256.
    chunk : int, optional
        The chunk number to load. The default is 0.

    Raises
    ------
    Exception
        if chunk is too high.

    Returns
    -------
    np_samples : TYPE
        DESCRIPTION.

    """
    audio = AudioSegment.from_file(file, format="mp3")
    length_ms = len(audio)
    size_whole_mb = length_ms/1000*sys.getsizeof(
        load_mp3_to_numpy(file,sec_end=1))/(1024**2)
    chunks_needed = roundup(size_whole_mb,limit_mb)
    if chunk >= chunks_needed:
        raise Exception(
            f"chunk {chunk} too high for chunks_needed {chunks_needed}")
    if chunks_needed==1:
        np_samples = load_mp3_to_numpy(file,sec_end=None)
    else:
        np_samples = load_mp3_to_numpy(file,
                                       sec_start=(length_ms/1000
                                                  *chunk/chunks_needed),
                                       sec_end=(length_ms/1000
                                                  *(chunk+1)/chunks_needed))
    return np_samples


def save_wav(np_samples,sampling_rate,name,folder=audio_save_folder):
    """
    A simple wrapper of scipy.io.wavfile.write.

    Parameters
    ----------
    np_samples : np.array of float, 1D
        The audio track to be saved.
    name : str
        The name of the file.
    sampling_rate : int
        The sampling rate in Hz.
    folder : str, optional
        The target folder. The default is paths.audio_save_folder.

    Returns
    -------
    None.

    """
    scaled = np.int16(np_samples / np.max(np.abs(np_samples)) * 32767)
    write(folder+name, sampling_rate, scaled)


#%%
if __name__ == "__main__":
    save_wav(load_mp3_to_numpy(audio_file),
             rate=get_sampling(audio_file),
             folder=audio_save_folder,
             name="test.wav")