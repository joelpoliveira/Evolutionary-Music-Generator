import os
import click
import librosa
import pandas as pd
import sounddevice as sd

from util.ga import *
from midiutil import MIDIFile
from datetime import datetime
from midi2audio import FluidSynth
from util.features import get_feature_vector
from scipy.spatial.distance import euclidean, cosine
#from util.features import get_feature_vector

FONT = "./soundfonts/piano_eletro.sf2"
MIDI_FILE = "./midi_files/"
WAV_FILE = "./wav_files/"
FILE_BASE_NAME = "ga-melody"

def euclidean_similarity(file_name: str, target_song_feat: ArrayLike):
    song, _ = librosa.load( get_wav(file_name) )
    feat = get_feature_vector(song)

    euc_dist = euclidean(feat, target_song_feat)
    euc_sim = 1 - euc_dist

    return euc_sim


def cosine_similarity(file_name: str, target_song_feat: ArrayLike):
    song, _ = librosa.load( get_wav(file_name) )
    feat = get_feature_vector(song)

    c_dist = cosine(feat, target_song_feat)
    c_sim = 1 - c_dist

    return c_sim


def flush():
    for file in os.listdir(MIDI_FILE):
        os.remove(MIDI_FILE + file)

    for file in os.listdir(WAV_FILE):
        os.remove(WAV_FILE + file)

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
    try:
        os.remove( get_midi(file_name) )
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

def chromosome_to_melody(chromosome: Chromosome, file_name: str) -> MIDIFile:
    midi_file = MIDIFile(1, deinterleave=False)
    melody = 0

    _, _, tempo, notes = split_chromosome(chromosome)
    midi_file.addTempo(1, 0, tempo)

    cumulative_time = 0
    for element in notes:
        duration, volume, note = element
        midi_file.addNote(
            track = melody, 
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

fitness: Callable


# notes = np.ravel(list(
#             map(lambda note: note[1], map(
#                     lambda c: c[2:], population
#                 )
#             )
#         ))
#         print(np.any(  (notes>127) ))
@click.command()
@click.option("--population-size", default=100, prompt='Population size:', type=int)
@click.option("--min-note", default=60, prompt='Lower MIDI possible value:', type=int)
@click.option("--max-note", default=81, prompt='Higher MIDI possible value:', type=int)
@click.option("--n-start", default=5, prompt="Number of starting notes", type=int)
@click.option("--n_gen", default = 50, prompt="Max. Number of Generations", type=int)
@click.option("--fit-func", default="euclidean", prompt="Similarity Metric", type=click.Choice(["euclidean", "cosine"]))
@click.option("--target", default="./jingles/Happy-birthday-piano-music.wav", prompt="WAV File for Fitness Score", type=str)
def main(
    population_size: int,
    min_note: int,
    max_note: int,
    n_start: int,
    n_gen: int,
    fit_func: str,
    target: str):
    
    try:
        song, _ = librosa.load(target, sr=44100)
        song_feat = get_feature_vector(song)
    except:
        print("Couldn't Open WAV file to calculate the fitness score!")
        print("Exiting")
        exit(-1)

    if fit_func=="euclidean":
        fitness = lambda file_name: euclidean_similarity( file_name, song_feat )
    else:
        fitness = lambda file_name: cosine_similarity( file_name, song_feat )

    population_gen = 0

    #generate initial population
    gen = Generator(min_note, max_note, n_start)
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

    while population_gen < n_gen: 
        print(f"Currently Processing Generation  {population_gen}")
        #generate respective file names
        file_names = [ FILE_BASE_NAME + f"{i}" for i in range(population_size)]
        
        
        #generate wav_files (nested loop is faster)
        print("\tGenerating Melodies")
        [ chromosome_to_melody(chromosome, fl) for chromosome, fl in zip(population, file_names) ]
        
        
        #calculate fitness for each chromosome
        print("\tCalculating Fitness")
        population_fitness = list(
                    map(lambda chromosome, file_name: (chromosome, fitness(file_name) ), population, file_names)
                )
        
        #generate statistics regarding fitness
        fitness_values = list(map(lambda pair: pair[1], population_fitness))
        max_fitness = np.max(fitness_values)
        avg_fitness = np.mean(fitness_values)
        std_fitness = np.std(fitness_values)

        #generate statistics regarding probabilities
        oper_probs=list(map(lambda c: c[0], population))
        mean_oper_probs = np.mean(oper_probs, axis=0)
        std_oper_probs = np.std(oper_probs, axis=0)

        mut_probs = list(map(lambda c: c[1], population))
        mean_mut_probs = np.mean(mut_probs, axis=0)
        std_mut_probs = np.std(mut_probs, axis=0)
        
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
        print(f"\t Best fitness obtained: {max_fitness}")

        #create the new generation
        population = selection(population_fitness, gen, population_size)

    save_data(statistics)
    flush()
if __name__ == '__main__':
    main()