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

FONT = "./soundfonts/piano_chords.sf2"
MIDI_FILE = "./midi_files/"
WAV_FILE = "./wav_files/"
FILE_BASE_NAME = "ga-melody"
NFFT=4096


def euclidean_similarity(feat: ArrayLike, target_song_feat: ArrayLike):
    if np.any(np.isnan(feat)) or np.any(np.isinf(feat)):
        print("THERE ARE NANS: ", feat)
        euc_sim = -np.inf
    else:
        euc_dist = euclidean(feat, target_song_feat)
        euc_sim = 1 - euc_dist

    return euc_dist


def cosine_similarity(feat: ArrayLike, target_song_feat: ArrayLike):
    if np.any(np.isnan(feat)) or np.any(np.isinf(feat)):
        c_sim = -np.inf
    else:
        c_dist = cosine(feat, target_song_feat)
        c_sim = 1 - c_dist

    return c_sim


def melody_to_feature_vector(N):
    v = []
    for i in range(N):
        song, fs = librosa.load( get_wav(FILE_BASE_NAME + f"-{i}"), sr=44100)
        v.append(get_feature_vector(song, NFFT))

    return v


def min_max_scale(y):
    min_v = y.min()
    max_v = y.max()

    if (max_v - min_v) == 0:
        return 0
    return (y - min_v) / (max_v - min_v)

def normalize(population_feat, song_feat):
    data = [song_feat] + population_feat
    norm_data = np.apply_along_axis(min_max_scale, 0, data)

    song_feat = norm_data[0, :]
    population_feat = norm_data[1:, :]

    return population_feat, song_feat


def flush():
    for file in os.listdir(MIDI_FILE):
        os.remove(MIDI_FILE + file)

    for file in os.listdir(WAV_FILE):
        os.remove(WAV_FILE + file)

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
        print("ERROR")
    return midi_file

fitness: Callable
song_feat_norm: ArrayLike

@click.command()
@click.option("--population-size", default=100, prompt='Population size:', type=int)
@click.option("--n_gen", default = 50, prompt="Max. Number of Generations", type=int)
@click.option("--min-note", default=21, prompt='Lower MIDI possible value:', type=int)
@click.option("--max-note", default=108, prompt='Higher MIDI possible value:', type=int)
@click.option("--n-start", default=5, prompt="Number of starting notes", type=int)
@click.option("--n-lines", default=2, prompt="Number of lines", type=int)
@click.option("--fit-func", default="euclidean", prompt="Similarity Metric", type=click.Choice(["euclidean", "cosine"]))
@click.option("--target", default="./jingles/happy-birthday.wav", prompt="WAV File for Fitness Score", type=str)
def main(
    population_size: int,
    n_gen: int,
    min_note: int,
    max_note: int,
    n_start: int,
    n_lines: int,
    fit_func: str,
    target: str):
    
    try:
        song, _ = librosa.load(target, sr=44100)
        song_feat = get_feature_vector(song, NFFT)
    except Exception as e:
        print(e)
        print("Couldn't Open WAV file to calculate the fitness score!")
        print("Exiting")
        exit(-1)

    if fit_func=="euclidean":
        fitness = lambda current_feat_norm: euclidean_similarity( current_feat_norm, song_feat_norm )
    else:
        fitness = lambda current_feat_norm: cosine_similarity( current_feat_norm, song_feat_norm )

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

    while population_gen < n_gen: 
        print(f"Currently Processing Generation  {population_gen}")
        
        #generate melodies
        print("\t..Generating Melodies..")
        [ chromosome_to_melody(c, FILE_BASE_NAME + f"-{i}") for i, c in enumerate(population) ]

        print("\t..Processing Features..")
        population_feat = melody_to_feature_vector(population_size)
        population_feat_norm, song_feat_norm = normalize(population_feat, song_feat)

        fitness_values = np.array([ fitness(feat) for feat in population_feat_norm ])
        population_fitness = list(zip(population, fitness_values))
        oper_probs = list(map(lambda c: c[0], population))
        
        print("\t..Extracting Statistics..")
        #get fitness statistics
        index = ( ~np.isnan(fitness_values)) | ( ~np.isinf(fitness_values))
        mean_fitness = fitness_values[ index ].mean()
        fitness_values[ ~index ] = mean_fitness
        max_fitness = np.max(fitness_values)
        std_fitness = np.std(fitness_values)

        #select best chromosome
        best_index = np.argmax(fitness_values)
        best_chromosome = population[best_index]
        best_individuals.append(best_chromosome)

        #get probabilities statistics
        mean_oper_probs = np.mean(oper_probs, axis=0)
        std_oper_probs = np.std(oper_probs, axis=0)
        
        population_gen += 1
        statistics.loc[population_gen] = {
            "Fitness_max" : max_fitness,
            "Fitness_mean" : mean_fitness,
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
        print(f"\t Best fitness obtained: {fitness_values[best_index]}")
        print(f"\t Mean fitness obtained: {mean_fitness}")

        #create the new generation
        population = selection(population_fitness, gen, population_size)
    save_data(statistics, best_individuals)
    flush()
if __name__ == '__main__':
    main()