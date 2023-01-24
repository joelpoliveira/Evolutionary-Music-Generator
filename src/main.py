import click
import pandas as pd

from util.ga import *
from util.audio_processor import *


def fitness(melody_file_name: str) -> Fitness:
    max_rate = 10; min_rate = 0
    t = PlaySong(melody_file_name, daemon=True)
    t.start()
    rating = input(f"Rating ({min_rate}-{max_rate}): ")
    
    t.stop()
    t.join()
    
    try:
        rating = float(rating)

        if rating>max_rate: rating = max_rate
        if rating<min_rate: rating = min_rate
    except ValueError:
        rating = 0
    return rating


@click.command()
@click.option("--population-size", default=10, prompt='Population size:', type=int)
@click.option("--min-note", default=0, prompt='Lower MIDI possible value:', type=int)
@click.option("--max-note", default=127, prompt='Higher MIDI possible value:', type=int)
@click.option("--n-start", default=2, prompt="Number of starting notes", type=int)
def main(
    population_size: int,
    min_note: int,
    max_note: int,
    n_start: int):
    
    running = True
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

        running = input(f"Finished generation {population_gen}.\nContinue to next generation? [Y/n]") != "n"
        
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