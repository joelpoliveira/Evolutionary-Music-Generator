import sounddevice as sd
import numpy as np
import librosa
import os

from util.audio_processor import *

FONT = "./soundfonts/piano_chords.sf2"



if __name__=="__main__":
    chromosome =[np.array([0.5663509 , 0.10387235, 0.12501908, 0.20475767]), np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]), 114.125, (4, 84, 85), (0.5, 84, 83), (1, 84, 81), (1.5, 84, 79), (1.5, 85, 77), (0.5, 85, 76), (1.5, 86, 74), (1, 86, 72), (1, 87, 71), (1, 87, 69), (1.5, 88, 67), (0.5, 88, 65), (1.5, 85, 77), (0.5, 85, 76), (1.5, 86, 74), (1, 86, 72), (1, 87, 71), (1, 87, 69), (1.5, 88, 67), (0.5, 88, 65), (1, 88, 64), (1, 89, 62), (1, 89, 60), (0.5, 90, 59), (1, 90, 57), (1.5, 91, 55), (1.5, 91, 53), (1, 92, 52), (1, 92, 50), (2, 106, 48)]
    fl = "temp"
    chromosome_to_melody(chromosome, fl)
    song, fs = librosa.load( get_wav(fl), sr=44100)

    sd.play(song, fs)
    sd.wait()
    
    os.remove( get_midi(fl) )
    os.remove( get_wav(fl) )


