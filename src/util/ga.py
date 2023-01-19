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
Population_Fitness = NewType("Population_and_Fitness", list[ Tuple[ Chromosome, Fitness ] ])

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
        Each note distances in scale from itself by twelve digits (in MIDI).
        For example: C1 is 24 and C2 is 36)

        This function randomly generates a 'base' note between MIN_NOTE and MAX_NOTE. 
        From this base note, the other notes can be selected by relative distance.
        """
        pitch_base = list(
                range(self._MIN_NOTE, self._MAX_NOTE, 12)
            )
        
        pitch = choice(pitch_base)
        return pitch
    
    def generate_random_note(self) -> Note:
        """
        A random note from C-major scale is selected (in the range 1 to 12).
        Given a base note (see 'generate_random_pitch'), the actual note is the
        increment of the base note with the firstly selected note.
        """
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
    """
    Separates the chromossome in 'Operator Probability', 'Mutation Probability' and 'Notes'
    """
    return chromosome[0], chromosome[1], chromosome[2:]


def tournament(
    population_fitness: Population_Fitness, 
    k: int 
) -> Chromosome:
    """
    Given a list of pairs (chromossome, fitness), performs a K-Tournament Selection,
    ordering the fitness in ascending order.
    """
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
    """
    Given a list of pairs (chromossome, fitness), removes the 10% elements with worse performance
    """
    to_remove = max(N//10, 1)
    population_fitness = population_fitness[:-to_remove]
    
    return population_fitness


def elitism(
    population_fitness: Population_Fitness,
    N: int
) -> Population:
    """
    Given a list of pairs (chromossome, fitness), selects the 10% elements with better results,
    in order to keep them in the next generation. 
    """
    to_keep = max(N//10, 1)
    new_population = population_fitness[:to_keep]
    
    return list(map(lambda x: x[0], new_population))


def selection(population_fitness: Population_Fitness) -> Population:
    """
    Given a list of pairs (chromossome, fitness), the selection method creates
    the new generation. Keeps best elements with 'Elitism' and generates new 
    population using one of the following operators per element:
        1. Mutation
        2. Crossover 
        3. Duplication
        4. Inversion
    Each operator as a probability of being selected.  
    """
    N = len(population_fitness)

    #TODO calculate fitness in main
    # population_fitness = list(map(lambda gen: (gen, fitness(gen)), population))

    population_fitness.sort(key = lambda x: x[1])
    population_fitness = truncate(population_fitness, N)
    new_population = elitism(population_fitness, N)
    
    k_left = N - len(new_population)
    while len(new_population) < N:
        chromosome = tournament(population_fitness, 2)
        
        operator = select_operator(chromosome, k_left)
        new_chromosomes = operator(chromosome, population_fitness, N) 

        new_population += new_chromosomes
        k_left-=len(new_chromosomes) 
        
    return new_population


def select_operator(
    chromosome: Chromosome,
    k_left: int
) -> Callable:
    """
    Given the 'Operations Probabilty' of the selected 'Chromosome'
    randomly selects the operation to be performed.
    """
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
    """
    Mutation Operator. 
    Given the 'Mutation Probabilities' of the Chromossome,a mutation 
    rule is selected randomly. 
    Then the selected mutation method is applied to the chromossome.
    """
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
    """
    Repeat Mutation Rule. 
    Randomly selects a note. That note is repeated 
    with the same duration.
    """
    prob_op, prob_mut, notes = split_chromosome(chromosome)
    n = random.choice(range(len(notes)))
    
    notes.insert(n, notes[n])
    chromosome = [prob_op, prob_mut, *notes]
    return chromosome


def split(chromosome: Chromosome) -> Chromosome:
    """
    Split Mutation Rule.
    Randomly selects a note. That note is split in two
    equal notes (each with half the duration of the original).
    """
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
    """
    Arpeggiate Mutation Rule.
    Randomly selects a note. That note is split in two notes
    with the duration of the original note. 
    The first note keeps it's pitch, the second note is transformed
    into a third (4 half-tones above) or a fith (7 half-tones above) 
    of the original pitch.
    Pitch is the numerical value of the note.
    """
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
    """
    Leap Mutation Rule.
    Randomly Selects a note. That note pitch is swaped to another
    note in the respective range (MIN_NOTE, MAX_NOTE).
    """
    prob_op, prob_mut, notes = split_chromosome(chromosome)
    n = random.choice(range(len(notes)))
    
    duration, note = notes[n]
    while (new := generator.generate_random_note())==note: continue
    
    new_note = (duration, new)
    notes[n] = new_note
    
    chromosome = [prob_op, prob_mut, *notes]
    return chromosome


def get_pairs(notes: list[Note]) -> list[int]:
    """
    Checks on which positions there are two sequential repeated notes.
    Returns an array with those indexes.
    """
    is_consecutive = []
    for i in range(len(notes)-1):
        _, note_a = notes[i]
        _, note_b = notes[i+1]
        
        if note_a==note_b:
            is_consecutive.append(i+1)
    return is_consecutive


def diatonic_upper_step_size(note: Note) -> int:
    """
    Given a note, checks wheter the next note in the diatonic scale 
    is half-step or a whole-step above.
    """
    if note%12 in (0, 2, 5, 7, 9):
        return 2
    return 1


def upper_neighbor(
    chromosome: Chromosome, 
    generator: Generator
) -> Chromosome:
    """
    Upper Neighbor Mutation Rule.
    Randomly selects a position where the same note is sequentially repeated.
    The second note is transposed to one diatonic scale step above itself.
    """
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
    """
    Given a note, checks wheter the previous note in the diatonic scale 
    is half-step or a whole-step below.
    """
    if note%12 in (0, 5):
        return 1
    return 2


def lower_neighbor(
    chromosome: Chromosome,
    generator: Generator
) -> Chromosome:
    """
    Lower Neighbor Mutation Rule.
    Randomly selects a position where the same note is sequentially repeated.
    The second note is transposed to one diatonic scale step below itself.
    """
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
    """
    Anticipation Mutation Rule.
    Randomly Selects a note. That note is split into two notes.
    The duration of the first is shorter than the second by the ratio 1:3.
    """
    prob_op, prob_mut, notes = split_chromosome(chromosome)
    n = random.choice(range(len(notes)))
    
    duration, note = notes[n]
    
    notes[n] = (duration * 0.25, note)
    notes.insert(n+1, (duration * 0.75, note))
    
    chromosome = [prob_op, prob_mut, *notes]
    return chromosome


def delay(chromosome: Chromosome) -> Chromosome:
    """
    Delay Mutation Rule.
    Randomly Selects a note. That note is split into two notes.
    The duration of the first is longer than the second by the ratio 3:1.
    """
    prob_op, prob_mut, notes = split_chromosome(chromosome)
    n = random.choice(range(len(notes)))
    
    duration, note = notes[n]
    
    notes[n] = (duration * 0.75, note)
    notes.insert(n+1, (duration * 0.25, note))
    
    chromosome = [prob_op, prob_mut, *notes]
    return chromosome


def passing_tone(chromosome: Chromosome) -> Chromosome:
    """
    Passing Tone Mutation Rule.
    Randomly selects a pair of consecutive notes.
    New notes are added in between them creating a stepwise motion
    to connect them.
    The implementation uses diatonic neighbors in the connection.
    If the first note is higher than the second - Downward scalar motion.
    If the first note is lower than the second - Upward scalar motion.
    """
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
    """
    Delete Mutation Rule.
    Randomly Selects a note. That note is deleted.
    """
    prob_op, prob_mut, notes = split_chromosome(chromosome)
    n = random.choice(range(len(notes)))
    
    notes = notes.copy()
    notes.pop(n)
    
    chromosome = [prob_op, prob_mut, *notes]
    return chromosome


def merge_note(chromosome):
    """
    Merge Note Mutation Rule.
    Randomly selects a position where the same note is sequentially repeated.
    The notes are merged into a single note with the duration being the sum of
    both notes duration. The maximum duration that the note can have is 8.
    """
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
    """
    Crossover Operator.
    Performs Single Point Crossover. This crossover cuts the
    chromossomes in a certain time point. It may cut notes 'in half'.
    """
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
    """
    Given a 'chromosome', randomly selects a time point in which 
    the chromosome will be splitten.
    """
    _, _, notes = split_chromosome(chromosome)
    duration = sum( note[0] for note in notes )
    
    point = random.choice(range(1, duration))
    return point


def cut_chromosome(
    chromosome: Chromosome,
    k: int
) -> tuple[Chromosome, Chromosome]:
    """
    Given a 'chromosome' and a time point, splits the chromosome
    in two parts. One before 'k' and another after 'k'.
    """
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
    """
    Given two cromossomes joins them.
    The notes are concatenated. The probabilities are averaged.
    """
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
    """
    Given a 'chromosome', randomly selects an interval in the 
    list of notes. That interval is duplicated.
    """
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
    """
    Given a 'chromosome', randomly selects an interval in the
    list of notes. That interval is reversed.
    """
    prob_op, prob_mut, notes = split_chromosome(chromosome)
    
    i = random.choice( range(len(notes) - 1) )
    j = random.choice( range(i+1, len(notes) ) )
                      
    to_reverse = notes[i : j]
    left = notes[:i]
    right = notes[j:]
    
    notes = left + to_reverse[::-1] + right
    
    chromosome = [prob_op, prob_mut, *notes]
    return [chromosome]