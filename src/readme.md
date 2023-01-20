# Description

Implementation of Genetic Algorithm to evolve melodies using a single part. It has two available methodologies.
    - 1. Interactively - The user ears the melodies and rates them. The melodies evolve according to the feedback they received from the user
    - 2. Automatically - The user presents a WAV file with a melody. Specifies the maximum number of iterations. The generated melodies evolve towards being similar to the input audio.

# Requirements
Python needs to be installed (version 3.9 or higher).

## 1. Python Packages

There are a few packages needed in order to run the project.
These packages are in the file 'requirements.txt'. 

In a console, walk  to the file directory.

If using python pip, type:

```console
$ pip install -r requirements.txt
```

If using python in a conda environment, type:

```console
$ conda install --file requirements.txt
```

## 2. Python Package Dependencies

The packages 'librosa' and 'midi2audio' use programs that need to be installed in the computer.

'midi2audio' uses Fluidsynth in order to convert MIDI files into WAV files. 
It can be downloaded from the console. The instructions are in their [official github repository](https://github.com/FluidSynth/fluidsynth/wiki/Download).

'librosa' uses FFmpeg to load audio files.
It can be downloaded from their [official site](https://ffmpeg.org/download.html).

Both programs bin folders need to be added to PATH.

# Execute program

In the 'src' folder simply run 'main.py':

```console
$ python main.py
```
