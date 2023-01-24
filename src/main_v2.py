import click
import pandas as pd

from util.ga import *
from util.audio_processor import *
from util.melodic_rules import get_score
#from util.features import get_feature_vector

FONT = "./soundfonts/piano_chords.sf2"
MIDI_FILE = "./midi_files/"
WAV_FILE = "./wav_files/"
FILE_BASE_NAME = "ga-melody"

def fitness(chromosome):
    return get_score(chromosome)

@click.command()
@click.option("--population-size", default=1000, prompt='Population size:', type=int)
@click.option("--n_gen", default = 50, prompt="Max. Number of Generations", type=int)
@click.option("--min-note", default=0, prompt='Lower MIDI possible value:', type=int)
@click.option("--max-note", default=127, prompt='Higher MIDI possible value:', type=int)
@click.option("--n-start", default=1, prompt="Number of starting notes", type=int)
def main(
    population_size: int,
    n_gen: int,
    min_note: int,
    max_note: int,
    n_start: int):

    population_gen = 0
    best_individuals = []
    worst_individuals = []
    #generate initial population
    gen = Generator(min_note, max_note, n_start, n_lines=1)
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

        fitness_values = np.array([ fitness(c) for c in population ])
        population_fitness = list(zip(population, fitness_values))
        oper_probs = list(map(lambda c: c[0], population))
        
        print("\t..Extracting Statistics..")
        #get fitness statistics
        mean_fitness = fitness_values.mean()
        max_fitness = np.max(fitness_values)
        std_fitness = np.std(fitness_values)

        #select best chromosome
        best_index = np.argmax(fitness_values)
        best_chromosome = population[best_index]
        best_individuals.append(best_chromosome)

        #select worst chromosome
        worst_index = np.argmin(fitness_values)
        worst_chromosome = population[worst_index]
        worst_individuals.append(worst_chromosome)



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
    save_data(statistics, best_individuals, worst_individuals)

    flush()
if __name__ == '__main__':
    main()