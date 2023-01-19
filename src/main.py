import os
from typing import Type
from scipy.io import wavfile
from midi2audio import FluidSynth
from midiutil import MIDIFile
from io import BytesIO
from util.features import get_feature_vector
from util.ga import *
import sounddevice as sd
import librosa
from threading import Thread

FONT = "./src/soundfonts/guitar.sf2"
MIDI_FILE = "./src/input/temp"
WAV_FILE = "./src/output/temp"

""" 
FONT = "piano_chords.sf2"
#MIDI_FILE = "major-scale.midi"
MIDI_FILE = "temp.midi"

WAV_FILE = "major-scale.wav"

degrees  = [60, 62, 64, 65, 67, 69, 71, 72] # MIDI note number
degrees = [4, 6, 8, 9, 11, 13, 15, 16]
track    = 0
channel  = 0
time     = 0   # In beats
duration = 1   # In beats
tempo    = 100  # In BPM
volume   = 100 # 0-127, as per the MIDI standard


MyMIDI = MIDIFile(1) # One track, defaults to format 1 (tempo track
                     # automatically created)
MyMIDI.addTempo(track,time, tempo)

for pitch in degrees:
    MyMIDI.addNote(track, channel, pitch, time, duration, volume)
    time = time + 1

with open(f"./input/{MIDI_FILE}", "wb") as output_file:
    MyMIDI.writeFile(output_file)

song = BytesIO()
wav_song = BytesIO( )

MyMIDI.writeFile(song)
FluidSynth(
        sound_font=f"./soundfonts/{FONT}"
    ).midi_to_audio(f"./input/{MIDI_FILE}", f"./output/{WAV_FILE}")

fs, song = wavfile.read(f"./output/{WAV_FILE}")
print(song)
sd.play(song, fs)
#sd.wait()
# sd.play()
#print(get_feature_vector(song)) """
""" 

def loop_play(file=None, sr=44100):
    t = Thread(target=play)
    t.start()

def play(file=None, sr=44100):
    if file == None: file=WAV_FILE+".wav"
    song, fs = librosa.load(file,sr=sr)
    sd.play(song, fs, loop=True)
    sd.wait()

def rand(low=0, high=2):
    return random.random()*(high - low + 1) + low


def save_harmony(midi_file, out):
    with open(MIDI_FILE + out + ".midi", "wb") as output_file:
        midi_file.writeFile(output_file)
        output_file.close()
    
    song = BytesIO()
    wav_song = BytesIO()
    
    midi_file.writeFile(song)

    FluidSynth(
            sound_font=FONT
        ).midi_to_audio(MIDI_FILE + out + ".midi", WAV_FILE + out + ".wav")   

 """


@click.command()
@click.option("--population-size", default=10, prompt='Population size:', type=int)
def main(population_size: int):
    
    running = True
    gen = Generator()
    population = Population([population_size])
    population_fitness = Population_Fitness([])
    population_gen = 0

    #avg fitness of each generation, statistics
    gen_avgFitness = [float]

    while running: 
        sum_fitness_gen = 0

        # Generate a number of random chromosomes
        for i in range(population_size):
            chromosome = gen.generate_random_chromosome()
            population.append(chromosome)

            #play and get fitness5
            midi = chromossome_to_melody(chromosome,"temp")
            print(midi)
            fitness = fitness_fun(chromosome)
            population_fitness.append((chromosome, fitness))
            sum_fitness_gen += fitness

        print(population_fitness)
        print(sum_fitness_gen/population_size)
        gen_avgFitness.append(sum_fitness_gen/population_size)

        running = input("Continue to next gen? [Y/n]") != "n"
        #create the new generation
        population = selection(population_fitness)
        population_gen += 1

if __name__ == '__main__':
    main()