import os
import librosa
import pandas as pd
import sounddevice as sd

from threading import Thread
from midiutil import MIDIFile
from datetime import datetime
from midi2audio import FluidSynth
from util.ga import split_chromosome

FONT = "./soundfonts/piano_eletro.sf2"
MIDI_FILE = "./midi_files/"
WAV_FILE = "./wav_files/"
FILE_BASE_NAME = "ga-melody"

class PlaySong(Thread):
    def __init__(self, file_name, *args, **kwargs):
        super(PlaySong, self).__init__(*args, **kwargs)

        song, fs = librosa.load(get_wav(file_name), sr = 44100)
        self.song = song
        self.fs = fs
        self.running = True

    def run(self):
        while self.running:
            sd.play(self.song, self.fs)
            sd.wait()

    def stop(self):
        self.running=False
        sd.stop()



def get_midi(fl_name: str)-> str:
    return MIDI_FILE + fl_name + ".mid"


def get_wav(fl_name: str)->str:
    return WAV_FILE + fl_name + ".wav"


def save_melody(midi : MIDIFile, file_name: str):
    """
    Saves the MIDI file as a '.mid' file. 
    Afterwards converts that file into a WAV file and saves it too.
    """
    try:
        #os.remove( get_midi(file_name) )
        with open( get_midi(file_name), "wb") as output_file:
            midi.writeFile(output_file)
            output_file.close()

        FluidSynth(
            sound_font=FONT
        ).midi_to_audio( get_midi(file_name), get_wav(file_name))
        return 1
    except Exception as e:
        print(e)
        print()
        return 0


def flush():
    for file in os.listdir(MIDI_FILE):
        os.remove(MIDI_FILE + file)

    for file in os.listdir(WAV_FILE):
        os.remove(WAV_FILE + file)


def save_data(df: pd.DataFrame, best_individuals: list, *args):
    folder = "./data/" + datetime.now().strftime("%d-%m-%Y %H-%M-%S")
    os.makedirs(folder)

    df.to_csv(folder + "/statistics.csv", index=True, index_label="Generation_ID")

    best_individuals_lines = list(map(lambda c: str(c).replace("\n", " "), best_individuals))
    with open(folder + "/individuals.txt", "w") as out:
        out.writelines(best_individuals_lines)
        out.close()

    if len(args)>0:
        worst_individuals_lines = list(map(lambda c: str(c).replace("\n", " "), args[0]))
        with open(folder + "/individuals_worst.txt", "w") as out:
            out.writelines(worst_individuals_lines)


def chromosome_to_melody(chromosome, file_name: str) -> MIDIFile:
    _, _, tempo, notes = split_chromosome(chromosome)

    midi_file = MIDIFile(
        1, 
        deinterleave=False, 
        adjust_origin=True,
        removeDuplicates=False)
    
    midi_file.addTempo(0, 0, tempo)
    cumulative_time = 0

    for element in notes:
        try:
            duration, volume, note = element
        except:
            print(element)
        midi_file.addNote(
            track = 0, 
            channel = 0, 
            pitch = note, 
            time = cumulative_time, 
            duration=duration/2, 
            volume=volume,
        )
        cumulative_time+=duration/2
        
    if save_melody(midi_file, file_name) == 0:
        print(chromosome[2:])
    return midi_file
