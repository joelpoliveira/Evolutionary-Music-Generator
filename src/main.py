import os
import click
import librosa
import pandas as pd
import sounddevice as sd

from util.ga import *
from threading import Thread
from midiutil import MIDIFile
from datetime import datetime
from midi2audio import FluidSynth

#from util.features import get_feature_vector

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

def save_data(df: pd.DataFrame):
    folder = "./data/" + datetime.now().strftime("%d-%m-%Y %H-%M-%S")
    os.makedirs(folder)

    df.to_csv(folder + "/statistics.csv", index=True, index_label="Generation_ID")

def get_midi(fl_name: str)-> str:
    return MIDI_FILE + fl_name + ".mid"

def get_wav(fl_name: str)->str:
    return WAV_FILE + fl_name + ".wav"

def save_melody(midi : MIDIFile, file_name: str):
    """
    Saves the MIDI file as a '.mid' file. 
    Afterwards converts that file into a WAV file and saves it too.
    """
    with open( get_midi(file_name), "wb") as output_file:
       midi.writeFile(output_file)
       output_file.close()

    FluidSynth(
        sound_font=FONT
    ).midi_to_audio( get_midi(file_name), get_wav(file_name))

def chromosome_to_melody(chromosome: Chromosome, file_name: str) -> MIDIFile:
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
            track = melody, 
            channel = 0, 
            pitch = note, 
            time = cumulative_time, 
            duration=duration/2, 
            volume=volume
        )

        cumulative_time+=duration/2
    save_melody(midi_file, file_name)
    return midi_file

def fitness(melody_file_name: str) -> Fitness:
    t = PlaySong(melody_file_name, daemon=True)
    t.start()
    rating = input("Rating (0-5): ")
    
    t.stop()
    t.join()
    
    try:
        rating = float(rating)

        max_rate = 5; min_rate = 0
        if rating>max_rate: rating = max_rate
        if rating<min_rate: rating = min_rate
    except ValueError:
        rating = 0
    return rating

@click.command()
@click.option("--population-size", default=10, prompt='Population size:', type=int)
@click.option("--min-note", default=60, prompt='Lower MIDI possible value:', type=int)
@click.option("--max-note", default=81, prompt='Higher MIDI possible value:', type=int)
@click.option("--n-start", default=5, prompt="Number of starting notes", type=int)
def main(
    population_size: int,
    min_note: int,
    max_note: int,
    n_start: int):
    
    running = True
    population_gen = 0

    #generate initial population
    gen = Generator(min_note, max_note, n_start)
    population = [ gen.generate_random_chromosome() for _ in range(population_size) ]

    #avg fitness of each generation, statistics
    statistics = pd.DataFrame(
        columns=["Max_fitness", "Avg_fitness", "Std_dev_fitness"]
    )

    

    while running: 
        
        #generate respective file names
        file_names = [ FILE_BASE_NAME + f"{i}" for i in range(population_size)]

        #generate wav_files (nested loop is faster)
        [ chromosome_to_melody(chromosome, fl) for chromosome, fl in zip(population, file_names) ]

        #calculate fitness for each chromosome
        population_fitness = list(
                    map(lambda chromosome, file_name: (chromosome, fitness(file_name) ), population, file_names)
                )
        
        #generation statistics
        fitness_values = list(map(lambda pair: pair[1], population_fitness))
        max_fitness = np.max(fitness_values)
        avg_fitness = np.mean(fitness_values)
        std_fitness = np.std(fitness_values)
        
        running = input("Continue to next gen? [Y/n]") != "n"
        
        population_gen += 1
        statistics.loc[population_gen] = {
            "Max_fitness" : max_fitness,
            "Avg_fitness" : avg_fitness,
            "Std_dev_fitness" : std_fitness
        }

        #create the new generation
        population = selection(population_fitness, gen, population_size)

    save_data(statistics)

if __name__ == '__main__':
    main()