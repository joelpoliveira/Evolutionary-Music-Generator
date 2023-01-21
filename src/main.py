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


def save_data(df: pd.DataFrame, best_individuals: list[Chromosome]):
    folder = "./data/" + datetime.now().strftime("%d-%m-%Y %H-%M-%S")
    os.makedirs(folder)

    df.to_csv(folder + "/statistics.csv", index=True, index_label="Generation_ID")

    best_individuals_lines = list(map(lambda c: str(c).replace("\n", " "), best_individuals))
    with open(folder + "/individuals.txt", "w") as out:
        out.writelines(best_individuals_lines)
        out.close()

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
    _, _, tempo, lines = split_lines(chromosome)

    midi_file = MIDIFile(
        len(lines), 
        deinterleave=False, 
        adjust_origin=True,
        removeDuplicates=False)
    
    for track, notes in enumerate(lines):
        midi_file.addTempo(track, 0, tempo)
        cumulative_time = 0

        for element in notes:
            duration, volume, note = element
            midi_file.addNote(
                track = track, 
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
@click.option("--min-note", default=36, prompt='Lower MIDI possible value:', type=int)
@click.option("--max-note", default=93, prompt='Higher MIDI possible value:', type=int)
@click.option("--n-start", default=5, prompt="Number of starting notes", type=int)
@click.option("--n-lines", default=2, prompt="Number of lines", type=int)
def main(
    population_size: int,
    min_note: int,
    max_note: int,
    n_start: int,
    n_lines: int):
    
    running = True
    population_gen = 0
    best_individuals = []
    #generate initial population
    gen = Generator(min_note, max_note, n_start, n_lines)
    population = [ gen.generate_random_chromosome() for _ in range(population_size) ]

    #avg fitness of each generation, statistics
    statistics = pd.DataFrame(
        columns=[
            "Fitness_max", "Fitness_mean", "Fitness_std", 
            "Mutation_mean", "Mutation_std",
            "Crossover_mean", "Crossover_std",
            "Duplication_mean", "Duplication_std",
            "Inversion_mean", "Inversion_std"
        ]
    )

    while running: 

        population_fitness = []
        fitness_values = []
        oper_probs = []
        mut_probs = []

        for i, chromosome in enumerate(population):
            file_name = FILE_BASE_NAME + f"-{i}"
            chromosome_to_melody(chromosome, file_name)
            current_fitness = fitness(file_name)
            fitness_values.append(current_fitness)
            population_fitness.append((chromosome, current_fitness))
            oper_probs.append(chromosome[0])
            mut_probs.append(chromosome[1])

        #select best chromosome
        best_index = np.argmax(fitness_values)
        best_chromosome = population[best_index]
        best_individuals.append(best_chromosome)

        #get fitness statistics
        max_fitness = np.max(fitness_values)
        avg_fitness = np.mean(fitness_values)
        std_fitness = np.std(fitness_values)
        
        #get probabilities statistics
        mean_oper_probs = np.mean(oper_probs, axis=0)
        std_oper_probs = np.std(oper_probs, axis=0)

        #mean_mut_probs = np.mean(mut_probs, axis=0)
        #std_mut_probs = np.std(mut_probs, axis=0)

        running = input("Continue to next gen? [Y/n]") != "n"
        
        population_gen += 1
        statistics.loc[population_gen] = {
            "Fitness_max" : max_fitness,
            "Fitness_mean" : avg_fitness,
            "Fitness_std" : std_fitness,
            "Mutation_mean" : mean_oper_probs[0],
            "Mutation_std" : std_oper_probs[0],
            "Crossover_mean" : mean_oper_probs[1],
            "Crossover_std" : std_oper_probs[1],
            "Duplication_mean" : mean_oper_probs[2],
            "Duplication_std": std_oper_probs[2],
            "Inversion_mean" : mean_oper_probs[3],
            "Inversion_std": std_oper_probs[3]
        }

        #create the new generation
        population = selection(population_fitness, gen, population_size)

    save_data(statistics, best_individuals)

if __name__ == '__main__':
    main()