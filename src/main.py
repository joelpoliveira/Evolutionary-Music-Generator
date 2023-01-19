import os
import click
import librosa
import sounddevice as sd

from util.ga import *
from threading import Thread
from midiutil import MIDIFile
from pyo import Server, SfPlayer
from midi2audio import FluidSynth
from util.features import get_feature_vector

FONT = "./soundfonts/piano_eletro.sf2"
MIDI_FILE = "./input/"
WAV_FILE = "./output/"
FILE_BASE_NAME = "ga-melody"


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

def get_midi(fl_name: str)-> str:
    return MIDI_FILE + fl_name + ".mid"

def get_wav(fl_name: str)->str:
    return WAV_FILE + fl_name + ".wav"

def save_melody(midi : MIDIFile, file_name: str):
    with open( get_midi(file_name), "wb") as output_file:
       midi.writeFile(output_file)
       output_file.close()

    FluidSynth(
        sound_font=FONT
    ).midi_to_audio( get_midi(file_name), get_wav(file_name))

def chromossome_to_melody(chromosome: Chromosome, file_name: str) -> MIDIFile:
    midi_file = MIDIFile(1)
    melody = 0

    tempo=120
    volume=100
    midi_file.addTempo(1, 0, tempo)

    _, _, notes = split_chromosome(chromosome)

    cumulative_time = 0
    for element in notes:
        duration, note = element
        midi_file.addNote(
            melody, 
            0, 
            note, 
            cumulative_time, 
            duration/2, 
            volume
        )

        cumulative_time+=duration/2
    save_melody(midi_file, file_name)
    return midi_file

def fitness_fun(melody_file_name: str) -> Fitness:
    
    server = Server(audio="offline_nb").boot()
    server.start()
    SfPlayer( get_wav(melody_file_name), speed=1, loop=True).out()
    rating = input("Rating (0-5)")
    server.stop()
    server.shutdown()
    try:
        rating = int(rating)
    except ValueError:
        rating = 0

    return rating


def play_melody(song, fs):
    sd.play(song, fs, loop=True)
    sd.wait()

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

            #play and get fitness
            chromossome_label = FILE_BASE_NAME + f"-{i}"
            chromossome_to_melody(chromosome, chromossome_label)

            fitness = fitness_fun(chromossome_label)
            population_fitness.append((chromosome, fitness))
            sum_fitness_gen += fitness

        #print(population_fitness)
        #print(sum_fitness_gen/population_size)
        gen_avgFitness.append(sum_fitness_gen/population_size)

        running = input("Continue to next gen? [Y/n]") != "n"
        #create the new generation
        population = selection(population_fitness)
        population_gen += 1

if __name__ == '__main__':
    main()