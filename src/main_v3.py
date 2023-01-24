import click
import librosa
import pandas as pd

from util.ga import *
from util.audio_processor import *
from util.features import get_feature_vector
from scipy.spatial.distance import euclidean, cosine

NFFT=4096
SR = 44100

def euclidean_similarity(feat: ArrayLike, target_song_feat: ArrayLike):
    if np.any(np.isnan(feat)) or np.any(np.isinf(feat)):
        print("THERE ARE NANS: ", feat)
        euc_sim = -np.inf
    else:
        euc_dist = euclidean(feat, target_song_feat)
        euc_sim = 1 - euc_dist

    return euc_sim


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
        song, fs = librosa.load( get_wav(FILE_BASE_NAME + f"-{i}"), sr=SR)
        v.append(get_feature_vector(song, NFFT, SR))

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




fitness: Callable
song_feat_norm: ArrayLike

@click.command()
@click.option("--population-size", default=100, prompt='Population size:', type=int)
@click.option("--n_gen", default = 50, prompt="Max. Number of Generations", type=int)
@click.option("--min-note", default=0, prompt='Lower MIDI possible value:', type=int)
@click.option("--max-note", default=127, prompt='Higher MIDI possible value:', type=int)
@click.option("--n-start", default=2, prompt="Number of starting notes", type=int)
@click.option("--fit-func", default="euclidean", prompt="Similarity Metric", type=click.Choice(["euclidean", "cosine"]))
@click.option("--target", default="./jingles/happy-birthday.wav", prompt="WAV File for Fitness Score", type=str)
def main(
    population_size: int,
    n_gen: int,
    min_note: int,
    max_note: int,
    n_start: int,
    fit_func: str,
    target: str):
    
    try:
        song, _ = librosa.load(target, sr=SR)
        song_feat = get_feature_vector(song, NFFT, SR)
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