# audio-bites
A playground to play with an audio track with numpy / scipy / cupy. There is no version without cuda, but you can simply comment the cupy import in edit.py and replace all "cp." with "np.", if your system is unable of running cupy. I actually don't know how much time cupy saves here, I expect a few seconds.

Your mp3 track is projected in the subspace of a few pre-difined chords. The motivation was to try to emulate vision with sound. The colors we see are a projection of a vector of freqency density of the incident light in the subspace of three spectra of transmission. 

Put the mp3 track of your choice to /music/mp3, change the path in paths.py, and run python edit.py. 
