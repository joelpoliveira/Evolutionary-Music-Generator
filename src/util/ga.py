import numpy as np

from random import *
from io import BytesIO
from midi2audio import FluidSynth
from numpy.typing import ArrayLike
from midiutil.MidiFile import MIDIFile
from typing import NewType, Union, Callable, Tuple

Note = NewType("Note", int)
Duration = NewType("Duration", int)
Chromosome = NewType("Chromosome", list[ 
    Union[ 
        ArrayLike, 
        Tuple[ 
            Duration, 
            Note
        ]
    ]
])
Population = NewType("Population", list[Chromosome])
Fitness = NewType("Fitness", float)
Population_Fitness = NewType("Population_and_Fitness", list[ tuple[ Chromosome, Fitness ] ])

class Generator:
    """
    Class Generator - Main class to access methods 
                        for generation of chromosomes 
                        or the corresponding phenotype

    Parameters: min_note -> Integer MIDI value for the lowest possible note. 
                            Kept as the attribute "_MIN_NOTE".
                max_note -> Integer MIDI value for the highest possible note
                            Kept as the attribute "_MAX_NOTE".
                n_starting_notes -> Number of notes to be generated.
                                    Kept as attribute "N".
    """
    def __init__(self, 
                min_note: int = 60,
                max_note: int = 81,
                n_starting_notes: int = 5):

        self._MIN_NOTE = min_note
        self._MAX_NOTE = max_note
        self._N = n_starting_notes
    
    def generate_random_pitch(self) -> Note:
        """
        
        """
        pitch_base = list(
                range(self._MIN_NOTE, self._MAX_NOTE, 12)
            )
        
        pitch = choice(pitch_base)
        return pitch
    
    def generate_random_note(self) -> Note:
        pitch = self.generate_random_pitch()
        
        # only notes from C-major scale
        notes = [1, 3, 5, 6, 8, 10, 12]
        note = choice(notes) + pitch
        return note

    def generate_random_duration(self) -> Duration:
        duration = choice(range(1,8+1))
        return duration

    def generate_random_chromosome(self) -> Chromosome:
        operator_probabilities= np.array([0.4, 0.2, 0.2, 0.2])
        mutation_probabilities = np.full(shape=11, fill_value=1/11)

        chromosome = [operator_probabilities, mutation_probabilities]
        for _ in range(self._N):
            duration = self.generate_random_duration()
            melody_note = self.generate_random_note()
            chromosome.append( (duration, melody_note) )
        return chromosome
    
# def chromossome_to_melody(chromosome: Chromosome, file_name: str) -> MIDIFile:
#     midi_file = MIDIFile(1)
#     melody = 0

#     tempo=120
#     volume=100
#     midi_file.addTempo(1, 0, tempo)

#     _, _, notes = split_chromosome(chromosome)

#     cumulative_time = 0
#     for element in notes:
#         duration, note = element
#         midi_file.addNote(
#             melody, 
#             0, 
#             note, 
#             cumulative_time, 
#             duration/2, 
#             volume
#         )

#         cumulative_time+=duration/2
#     #save_melody(chromosome, file_name)
#     return midi_file

# def save_melody(midi : MIDIFile, file_name: str):
#     with open(file_name + "mid", "wb") as output_file:
#        midi.writeFile(output_file)
#        output_file.close()

#     song = BytesIO()
#     midi.writeFile(song)
#     FluidSynth(
#         sound_font=FONT
#     ).midi_to_audio(file_name+".mid", file_name + ".wav")

def split_chromosome(chromosome):
    return chromosome[0], chromosome[1], chromosome[2:]


def tournament(
    population_fitness: Population_Fitness, 
    k: int 
) -> Chromosome:

    indexes = [i for i in range(len(population_fitness))]
    selected_indexes = list(np.random.choice(
        indexes, 
        size=k, 
        replace=False
    ))

    population_fitness_subset = [ population_fitness[i] for i in selected_indexes]
    population_fitness_subset.sort(key = lambda x: x[1])

    chromosome_selected = population_fitness_subset[0]
    return chromosome_selected[0]


def truncate(
    population_fitness: Population_Fitness,
    N : int
) -> Population_Fitness:

    to_remove = max(N//10, 1)
    population_fitness = population_fitness[:-to_remove]
    
    return population_fitness


def elitism(
    population_fitness: Population_Fitness,
    N: int
) -> Population:

    to_keep = max(N//10, 1)
    new_population = population_fitness[:to_keep]
    
    return list(map(lambda x: x[0], new_population))


def selection(population_fitness: Population_Fitness) -> Population:
    N = len(population_fitness)

    #TODO calculate fitness in main
    # population_fitness = list(map(lambda gen: (gen, fitness(gen)), population))

    population_fitness.sort(key = lambda x: x[1])
    population_fitness = truncate(population_fitness, N)
    new_population = elitism(population_fitness, N)
    
    k_left = N - len(new_population)
    while len(new_population) < N:
        chromosome = tournament(population_fitness, 2)
        
        probs_op, probs_mut, notes = split_chromosome(chromosome)
        operator = select_operator(chromosome, k_left)
        new_chromosomes = operator(chromosome, population_fitness, N) 

        new_population += new_chromosomes
        k_left-=len(new_chromosomes) 
        
    return new_population


def select_operator(
    chromosome: Chromosome,
    k_left: int
) -> Callable:
    prob_op, _, _ = split_chromosome(chromosome)
    if k_left>=2:
        operator_index = np.random.choice( range(4), size=1, p=prob_op )[0]
    else:
        available = [0, 2, 3]

        #probability needs to sum to 1
        temp_prob = prob_op[available]
        temp_prob /= temp_prob.sum()

        operator_index = np.random.choice( available, size=1, p=temp_prob)[0]
        
    operators = [ mutation, crossover, duplication, inversion ]
    return operators[ operator_index ]




########################################################
########################################################
######                                            ######
######  Functions/Procedures Related to Mutation  ######
######                                            ######
########################################################
########################################################



def mutation(chromosome: Chromosome, *args) -> list[Chromosome]:
    mutation_rules: list[Callable] = [
        repeat, split, arpeggiate, leap,
        upper_neighbor, lower_neighbor,
        anticipation, delay, passing_tone,
        delete_note, merge_note
    ]

    _, prob_mut, _ = split_chromosome(chromosome)
    rule = np.random.choice(
        mutation_rules,
        size=1,
        p=prob_mut
    )[0]

    chromosome = rule(chromosome)
    return [chromosome]


def repeat(chromosome: Chromosome) -> Chromosome:
    prob_op, prob_mut, notes = split_chromosome(chromosome)
    n = random.choice(range(len(notes)))
    
    notes.insert(n, notes[n])
    chromosome = [prob_op, prob_mut, *notes]
    return chromosome


def split(chromosome: Chromosome) -> Chromosome:
    prob_op, prob_mut, notes = split_chromosome(chromosome)
    n = random.choice(range(len(notes)))
    
    duration, note = notes[n]
    new_duration = duration/2
    
    new_note = (new_duration, note)
    notes[n] = new_note
    notes.insert(n, new_note)
    
    chromosome = [prob_op, prob_mut, *notes]
    return chromosome


def arpeggiate(chromosome: Chromosome, generator: Generator) -> Chromosome:
    prob_op, prob_mut, notes = split_chromosome(chromosome)
    n = random.choice(range(len(notes)))
    
    duration, note = notes[n]
    pitch_incr = random.choice([4,7])
    
    new_pitch = min(generator._MAX_NOTE, note+pitch_incr)
    new_note = (duration, new_pitch)
    
    notes.insert(n+1, new_note)
    
    chromosome = [prob_op, prob_mut, *notes]
    return chromosome


def leap(
    chromosome: Chromosome, 
    generator: Generator
) -> Chromosome:
    prob_op, prob_mut, notes = split_chromosome(chromosome)
    n = random.choice(range(len(notes)))
    
    duration, note = notes[n]
    while (new := generator.generate_random_note())==note: continue
    
    new_note = (duration, new)
    notes[n] = new_note
    
    chromosome = [prob_op, prob_mut, *notes]
    return chromosome


def get_pairs(notes: list[Note]) -> list[int]:
    is_consecutive = []
    for i in range(len(notes)-1):
        _, note_a = notes[i]
        _, note_b = notes[i+1]
        
        if note_a==note_b:
            is_consecutive.append(i+1)
    return is_consecutive


def diatonic_upper_step_size(note: Note) -> int:
    if note%12 in (0, 2, 5, 7, 9):
        return 2
    return 1


def upper_neighbor(
    chromosome: Chromosome, 
    generator: Generator
) -> Chromosome:

    prob_op, prob_mut, notes = split_chromosome(chromosome)
    
    is_consecutive = get_pairs(notes)
    if len(is_consecutive)==0:
        return chromosome
    
    n = random.choice(is_consecutive)
    d, note = notes[n]
    
    step = diatonic_upper_step_size(note)
    note = min(generator._MAX_NOTE, note+step)
    
    notes[n] = (d, note)
    chromosome = [prob_op, prob_mut, *notes]
    return chromosome 


def diatonic_lower_step_size(note: Note) -> int:
    if note%12 in (0, 5):
        return 1
    return 2


def lower_neighbor(
    chromosome: Chromosome,
    generator: Generator
) -> Chromosome:
    prob_op, prob_mut, notes = split_chromosome(chromosome)

    is_consecutive = get_pairs(notes)
    if len(is_consecutive)==0:
        return chromosome
    
    n = random.choice(is_consecutive)
    d, note = notes[n]
    
    step = diatonic_lower_step_size(note)
    note = max(generator._MIN_NOTE, note - step)
    
    notes[n] = (d, note)
    chromosome = [prob_op, prob_mut, *notes]
    return chromosome


def anticipation(chromosome: Chromosome) -> Chromosome:
    prob_op, prob_mut, notes = split_chromosome(chromosome)
    n = random.choice(range(len(notes)))
    
    duration, note = notes[n]
    
    notes[n] = (duration * 0.25, note)
    notes.insert(n+1, (duration * 0.75, note))
    
    chromosome = [prob_op, prob_mut, *notes]
    return chromosome


def delay(chromosome: Chromosome) -> Chromosome:
    prob_op, prob_mut, notes = split_chromosome(chromosome)
    n = random.choice(range(len(notes)))
    
    duration, note = notes[n]
    
    notes[n] = (duration * 0.75, note)
    notes.insert(n+1, (duration * 0.25, note))
    
    chromosome = [prob_op, prob_mut, *notes]
    return chromosome


def passing_tone(chromosome: Chromosome) -> Chromosome:
    prob_op, prob_mut, notes = split_chromosome(chromosome)
    n = random.choice( range( len( notes ) - 1 ) )
    
    d, note = notes[n]
    d_b, note_b = notes[n+1]
    
    d = min(d, d_b)/2
    if note>note_b:   
        new_notes = []
        while note - ( step := diatonic_lower_step_size(note) ) > note_b:
            note -= step
            new = (d, note)
            new_notes.append(new)
        notes = notes[:n+1] + new_notes + notes[n+1:]
        
    if note<note_b:
        new_notes = []
        while note + ( step := diatonic_upper_step_size(note) ) < note_b:
            note += step
            new = (d, note)
            new_notes.append(new)
        notes = notes[:n+1] + new_notes + notes[n+1:]
    chromosome = [prob_op, prob_mut, *notes]
    return chromosome   


def delete_note(chromosome: Chromosome) -> Chromosome:
    prob_op, prob_mut, notes = split_chromosome(chromosome)
    n = random.choice(range(len(notes)))
    
    notes = notes.copy()
    notes.pop(n)
    
    chromosome = [prob_op, prob_mut, *notes]
    return chromosome


def merge_note(chromosome):
    prob_op, prob_mut, notes = split_chromosome(chromosome)
    is_consecutive = get_pairs(notes)
    
    if len(is_consecutive)==0: 
        return chromosome
    
    n = random.choice(is_consecutive)
    d_a, note_a = notes[n-1]
    d_b, _ = notes[n]
    
    d = min(8, d_a + d_b)
    new = (d, note_a)
    
    notes[n-1] = new
    notes = notes.copy()
    
    notes.pop(n)
    chromosome = [prob_op, prob_mut, *notes]
    return chromosome


##########################################################
##########################################################
######                                             #######
######  Functions/Procedures Related to Crossover  #######
######                                             #######
##########################################################
##########################################################


def crossover(
    chromosome: Chromosome,
    population_fitness: Population_Fitness,
    N: int 
) -> list[Chromosome]:
    k = max(2, N//5)
    other_chromosome = tournament(population_fitness, k)
    
    i = get_cutpoint(chromosome)
    j = get_cutpoint(other_chromosome)
    
    chromosome_left, chromosome_right = cut_chromosome(chromosome, i)
    other_left, other_right = cut_chromosome(other_chromosome, j)
    
    chromosome_a = merge_chromosomes(chromosome_left, other_right)
    chromosome_b = merge_chromosomes(other_left, chromosome_left)
    
    return [chromosome_a, chromosome_b]


def get_cutpoint(chromosome: Chromosome) -> int:
    _, _, notes = split_chromosome(chromosome)
    duration = sum( note[0] for note in notes )
    
    point = random.choice(range(1, duration))
    return point


def cut_chromosome(
    chromosome: Chromosome,
    k: int
) -> tuple[Chromosome, Chromosome]:
    prob_op, prob_mut, notes = split_chromosome(chromosome)
    
    cum_time=0
    for i in range(len(notes)):
        d, note = notes[i]
        
        if cum_time == k:
            chromosome_left = [prob_op, prob_mut, *notes[:i]]
            chromosome_right = [prob_op, prob_mut, *notes[i:]]
            
            return chromosome_left, chromosome_right
        
        elif cum_time< k < cum_time+d:
            d_left = k - cum_time - 1
            d_right = d - d_left
            
            notes_left = notes[:i+1]
            notes_right = notes[i:]
            
            notes_left[-1] = (d_left, note)
            notes_right[0] = (d_right, note)
            
            chromosome_left = [ prob_op, prob_mut, *notes_left ]
            chromosome_right = [prob_op, prob_mut, *notes_right ]
            
            return chromosome_left, chromosome_right
        else:
            cum_time+=d


def merge_chromosomes( 
    left: Chromosome, 
    right: Chromosome 
) -> Chromosome:

    prob_op_left, prob_mut_left, notes_left = split_chromosome(left)
    prob_op_right, prob_mut_right, notes_right = split_chromosome(right)
    
    prob_op = ( prob_op_left + prob_op_right ) / 2
    prob_mut = ( prob_mut_left + prob_mut_right) / 2
    notes = notes_left + notes_right
    
    chromosome = [prob_op, prob_mut, *notes]
    return chromosome



############################################################
############################################################
######                                               #######
######  Functions/Procedures Related to Duplication  #######
######                                               #######
############################################################
############################################################


def duplication(chromosome: Chromosome, *args) -> list[Chromosome]:
    prob_op, prob_mut, notes = split_chromosome(chromosome)
    
    i = random.choice( range(len(notes)-1) )
    j = random.choice( range(1,8+1) )
        
    to_duplicate = notes[i : i+j]
    left = notes[ :i ]
    right = notes[ i+j: ]
    notes = left + to_duplicate + to_duplicate + right
    
    chromosome = [ prob_op, prob_mut, *notes]
    return [chromosome]


##########################################################
##########################################################
######                                             #######
######  Functions/Procedures Related to Inversion  #######
######                                             #######
##########################################################
##########################################################

def inversion(chromosome: Chromosome, *args) -> list[Chromosome]:
    prob_op, prob_mut, notes = split_chromosome(chromosome)
    
    i = random.choice( range(len(notes) - 1) )
    j = random.choice( range(i+1, len(notes) ) )
                      
    to_reverse = notes[i : j]
    left = notes[:i]
    right = notes[j:]
    
    notes = left + to_reverse[::-1] + right
    
    chromosome = [prob_op, prob_mut, *notes]
    return [chromosome]