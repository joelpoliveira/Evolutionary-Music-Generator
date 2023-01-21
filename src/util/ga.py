import numpy as np
from random import *
from numpy.typing import ArrayLike
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
                n_starting_notes: int = 5,
                n_lines = 2):

        self._MIN_NOTE = min_note
        self._MAX_NOTE = max_note
        self._N = n_starting_notes
        self.n_lines = n_lines
    
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

    def generate_random_tempo(self):
        return choice(range(60,140+1))

    def generate_random_volume(self):
        return choice(range(80, 110+1))

    def generate_random_duration(self) -> Duration:
        duration = choice(range(1,8+1))
        return duration

    def generate_random_chromosome(self) -> Chromosome:
        operator_probabilities= np.array([0.4, 0.2, 0.2, 0.2])
        mutation_probabilities = np.full(shape=11, fill_value=1/11)

        lines = []
        tempo = self.generate_random_tempo()
        for _ in range(self.n_lines):
            notes = []
            for __ in range(self._N):
                duration = self.generate_random_duration()
                melody_note = self.generate_random_note()
                volume = self.generate_random_volume()
                notes.append( (duration, volume, melody_note) )
            lines.append(notes)

        chromosome = [
            operator_probabilities,
            mutation_probabilities,
            tempo,
            lines
        ]
        return chromosome


def split_chromosome(chromosome):
    """
    Separates the line-chromossome in 'Operator Probability', 'Mutation Probability' and 'Notes'
    """
    return chromosome[0], chromosome[1], chromosome[2], chromosome[3:]

def split_lines(chromosome):
    """
    Separates the chromossome in 'Operator Probability', 'Mutation Probability' and 'Lines'
    """
    return chromosome[0], chromosome[1], chromosome[2], chromosome[3]


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


def selection(
    population_fitness: Population_Fitness, 
    generator: Generator,
    N: int
) -> Population:
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

    population_fitness.sort(key = lambda x: x[1], reverse=True)
    population_fitness = truncate(population_fitness, N)
    new_population = elitism(population_fitness, N)
    
    k_left = N - len(new_population)
    while len(new_population) < N:
        chromosome = tournament(population_fitness, 2)
        
        operator = select_operator(chromosome, k_left)
        new_chromosomes = operator(chromosome, population_fitness, N, generator=generator) 
        new_chromosomes = mutate_probabilities(new_chromosomes)
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
    prob_op, _, _, _ = split_lines(chromosome)
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



def mutation(chromosome: Chromosome, *args, **kwargs) -> list[Chromosome]:
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

    prob_op, prob_mut, tempo, lines = split_lines(chromosome)
    rule = np.random.choice(
        mutation_rules,
        size=1,
        p=prob_mut
    )[0]

    new_tempo = 0
    new_lines = []
    new_probs_op = np.zeros(4)
    new_probs_mut = np.zeros(11)
    i = 0
    for notes in lines:
        sub_chromosome = [prob_op, prob_mut, tempo, *notes]
        sub_chromosome = rule(sub_chromosome, **kwargs)

        new_probs_op += sub_chromosome[0]
        new_probs_mut += sub_chromosome[1]
        new_tempo += sub_chromosome[2]
        new_lines.append(sub_chromosome[3:])
        i+=1

    chromosome = [ new_probs_op/i, new_probs_mut/i, int(new_tempo/i), new_lines]
    return [chromosome]


def repeat(chromosome: Chromosome, **kwargs) -> Chromosome:
    """
    Repeat Mutation Rule. 
    Randomly selects a note. That note is repeated 
    with the same duration.
    """
    prob_op, prob_mut, tempo, notes = split_chromosome(chromosome)
    n = choice(range(len(notes)))
    
    notes.insert(n, notes[n])
    chromosome = [prob_op, prob_mut, tempo, *notes]
    return chromosome


def split(chromosome: Chromosome, **kwargs) -> Chromosome:
    """
    Split Mutation Rule.
    Randomly selects a note. That note is split in two
    equal notes (each with half the duration of the original).
    """
    prob_op, prob_mut, tempo, notes = split_chromosome(chromosome)
    n = choice(range(len(notes)))
    
    duration, volume, note = notes[n]
    new_duration = duration/2
    
    new_note = (new_duration, volume, note)
    notes[n] = new_note
    notes.insert(n, new_note)
    
    chromosome = [prob_op, prob_mut, tempo, *notes]
    return chromosome


def arpeggiate(chromosome: Chromosome, **kwargs) -> Chromosome:
    """
    Arpeggiate Mutation Rule.
    Randomly selects a note. That note is split in two notes
    with the duration of the original note. 
    The first note keeps it's pitch, the second note is transformed
    into a third (4 half-tones above) or a fith (7 half-tones above) 
    of the original pitch.
    Pitch is the numerical value of the note.
    """
    generator = kwargs["generator"]
    prob_op, prob_mut, tempo, notes = split_chromosome(chromosome)
    n = choice(range(len(notes)))
    
    duration, volume, note = notes[n]
    pitch_incr = choice([4,7])
    
    new_pitch = min(generator._MAX_NOTE, note+pitch_incr)
    new_note = (duration, volume, new_pitch)
    
    notes.insert(n+1, new_note)
    
    chromosome = [prob_op, prob_mut, tempo, *notes]
    return chromosome


def leap(
    chromosome: Chromosome, 
    **kwargs
) -> Chromosome:
    """
    Leap Mutation Rule.
    Randomly Selects a note. That note pitch is swaped to another
    note in the respective range (MIN_NOTE, MAX_NOTE).
    """
    generator = kwargs["generator"]
    prob_op, prob_mut, tempo, notes = split_chromosome(chromosome)
    n = choice(range(len(notes)))
    
    duration, volume, note = notes[n]
    while (new := generator.generate_random_note())==note: continue
    
    new_note = (duration, volume, new)
    notes[n] = new_note
    
    chromosome = [prob_op, prob_mut, tempo, *notes]
    return chromosome


def get_pairs(notes: list[Note]) -> list[int]:
    """
    Checks on which positions there are two sequential repeated notes.
    Returns an array with those indexes.
    """
    is_consecutive = []
    for i in range(len(notes)-1):
        _, _, note_a = notes[i]
        _, _, note_b = notes[i+1]
        
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
    chromosome: Chromosome, **kwargs
) -> Chromosome:
    """
    Upper Neighbor Mutation Rule.
    Randomly selects a position where the same note is sequentially repeated.
    The second note is transposed to one diatonic scale step above itself.
    """
    generator = kwargs["generator"]
    prob_op, prob_mut, tempo, notes = split_chromosome(chromosome)
    
    is_consecutive = get_pairs(notes)
    if len(is_consecutive)==0:
        return chromosome
    
    n = choice(is_consecutive)
    d, volume, note = notes[n]
    
    step = diatonic_upper_step_size(note)
    note = min(generator._MAX_NOTE, note+step)
    
    notes[n] = (d, volume, note)
    chromosome = [prob_op, prob_mut, tempo, *notes]
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
    chromosome: Chromosome, **kwargs
) -> Chromosome:
    """
    Lower Neighbor Mutation Rule.
    Randomly selects a position where the same note is sequentially repeated.
    The second note is transposed to one diatonic scale step below itself.
    """
    generator = kwargs["generator"]
    prob_op, prob_mut, tempo, notes = split_chromosome(chromosome)

    is_consecutive = get_pairs(notes)
    if len(is_consecutive)==0:
        return chromosome
    
    n = choice(is_consecutive)
    d, volume, note = notes[n]
    
    step = diatonic_lower_step_size(note)
    note = max(generator._MIN_NOTE, note - step)
    
    notes[n] = (d, volume, note)
    chromosome = [prob_op, prob_mut, tempo, *notes]
    return chromosome


def anticipation(chromosome: Chromosome, **kwargs) -> Chromosome:
    """
    Anticipation Mutation Rule.
    Randomly Selects a note. That note is split into two notes.
    The duration of the first is shorter than the second by the ratio 1:3.
    """
    prob_op, prob_mut, tempo, notes = split_chromosome(chromosome)
    n = choice(range(len(notes)))
    
    duration, volume, note = notes[n]
    
    notes[n] = (duration * 0.25, min(volume+5, 110), note)
    notes.insert(n+1, ( duration * 0.75, volume, note))
    
    chromosome = [prob_op, prob_mut, tempo, *notes]
    return chromosome


def delay(chromosome: Chromosome, **kwargs) -> Chromosome:
    """
    Delay Mutation Rule.
    Randomly Selects a note. That note is split into two notes.
    The duration of the first is longer than the second by the ratio 3:1.
    """
    prob_op, prob_mut, tempo, notes = split_chromosome(chromosome)
    n = choice(range(len(notes)))
    
    duration, volume, note = notes[n]
    
    notes[n] = ( duration * 0.75, volume, note)
    notes.insert(n+1, (duration * 0.25, min(volume+5, 110), note))
    
    chromosome = [prob_op, prob_mut, tempo, *notes]
    return chromosome


def passing_tone(chromosome: Chromosome, **kwargs) -> Chromosome:
    """
    Passing Tone Mutation Rule.
    Randomly selects a pair of consecutive notes.
    New notes are added in between them creating a stepwise motion
    to connect them.
    The implementation uses diatonic neighbors in the connection.
    If the first note is higher than the second - Downward scalar motion.
    If the first note is lower than the second - Upward scalar motion.
    """
    prob_op, prob_mut, tempo, notes = split_chromosome(chromosome)
    if len(notes)<=1:
        return chromosome
    n = choice( range( len( notes ) - 1 ) )
    
    d, volume, note = notes[n]
    d_b, volume_b, note_b = notes[n+1]
    
    
    vs = np.linspace(volume, volume_b)
    d =min(d, d_b)/2
    i=0

    if note>note_b:   
        new_notes = []
        while note - ( step := diatonic_lower_step_size(note) ) > note_b:
            note -= step
            new = (d, int(vs[i]), note)
            new_notes.append(new)
            i+=1

        notes = notes[:n+1] + new_notes + notes[n+1:]
        
    if note<note_b:
        new_notes = []
        while note + ( step := diatonic_upper_step_size(note) ) < note_b:
            note += step
            new = (d, int(vs[i]), note)
            new_notes.append(new)
            i+=1
            
        notes = notes[:n+1] + new_notes + notes[n+1:]
    chromosome = [prob_op, prob_mut, tempo, *notes]
    return chromosome   


def delete_note(chromosome: Chromosome, **kwargs) -> Chromosome:
    """
    Delete Mutation Rule.
    Randomly Selects a note. That note is deleted.
    """
    prob_op, prob_mut, tempo, notes = split_chromosome(chromosome)
    n = choice(range(len(notes)))
    
    notes = notes.copy()
    notes.pop(n)
    
    chromosome = [prob_op, prob_mut, tempo, *notes]
    return chromosome


def merge_note(chromosome, **kwargs):
    """
    Merge Note Mutation Rule.
    Randomly selects a position where the same note is sequentially repeated.
    The notes are merged into a single note with the duration being the sum of
    both notes duration. The maximum duration that the note can have is 8.
    """
    prob_op, prob_mut, tempo, notes = split_chromosome(chromosome)
    is_consecutive = get_pairs(notes)
    
    if len(is_consecutive)==0: 
        return chromosome
    
    n = choice(is_consecutive)
    d_a, volume_a, note_a = notes[n-1]
    d_b, volume_b, _ = notes[n]
    
    d = min(8, d_a + d_b)
    v = int((volume_a + volume_b) / 2)
    new = (d, v, note_a)
    
    notes[n-1] = new
    notes = notes.copy()
    
    notes.pop(n)
    chromosome = [prob_op, prob_mut, tempo, *notes]
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
    N: int, **kwargs
) -> list[Chromosome]:

    k = max(2, N//5)
    other_chromosome = tournament(population_fitness, k)
    
    prob_op_a, prob_mut_a, tempo_a, lines_a = split_lines(chromosome)
    prob_op_b, prob_mut_b, tempo_b, lines_b = split_lines(other_chromosome)

    new_lines_a = []
    new_probs_op_a = 0
    new_probs_mut_a = 0 
    new_tempo_a = 0

    new_lines_b = []
    new_probs_op_b = 0
    new_probs_mut_b = 0 
    new_tempo_b = 0

    i=0
    for notes_a, notes_b in zip(lines_a, lines_b):
        c_a = [prob_op_a, prob_mut_a, tempo_a, *notes_a]
        c_b = [prob_op_b, prob_mut_b, tempo_b, *notes_b]
        new_ca, new_cb = sub_crossover(c_a, c_b, N)

        temp_op, temp_mut, temp_tempo, temp_notes = split_chromosome(new_ca)
        new_probs_op_a = temp_op + new_probs_op_a
        new_probs_mut_a = temp_mut + new_probs_mut_a
        new_tempo_a+=temp_tempo
        new_lines_a.append( temp_notes)

        temp_op, temp_mut, temp_tempo, temp_notes = split_chromosome(new_cb)
        new_probs_op_b = temp_op + new_probs_op_b
        new_probs_mut_b =temp_mut + new_probs_mut_b
        new_tempo_b+=temp_tempo
        new_lines_b.append( temp_notes)

        i+=1

    chromosome_a = [ new_probs_op_a/i, new_probs_mut_a/i, int(new_tempo_a/i), new_lines_a]
    chromosome_b = [ new_probs_op_b/i, new_probs_mut_b/i, int(new_tempo_b/i), new_lines_b]

    return [chromosome_a, chromosome_b]

def sub_crossover(
    chromosome: Chromosome,
    other_chromosome: Chromosome,
    N: int, **kwargs
) -> list[Chromosome]:
    """
    Crossover Operator.
    Performs Single Point Crossover. This crossover cuts the
    chromossomes in a certain time point. It may cut notes 'in half'.
    """
    
    i = get_cutpoint(chromosome)
    j = get_cutpoint(other_chromosome)
    
    chromosome_left, chromosome_right = cut_chromosome(chromosome, i)
    other_left, other_right = cut_chromosome(other_chromosome, j)
    
    chromosome_a = merge_chromosomes(chromosome_left, other_right)
    chromosome_b = merge_chromosomes(other_left, chromosome_right)
    
    return [chromosome_a, chromosome_b]


def get_cutpoint(chromosome: Chromosome) -> int:
    """
    Given a 'chromosome', randomly selects a time point in which 
    the chromosome will be splitten.
    """
    _, _, _, notes = split_chromosome(chromosome)
    max_duration = int(sum( note[0] for note in notes ))
    if max_duration<=1:
        return 0

    point = choice(range(max_duration - 1))
    return point


def cut_chromosome(
    chromosome: Chromosome,
    k: int
) -> tuple[Chromosome, Chromosome]:
    """
    Given a 'chromosome' and a time point, splits the chromosome
    in two parts. One before 'k' and another after 'k'.
    """
    prob_op, prob_mut, tempo, notes = split_chromosome(chromosome)
    
    cum_time=0
    for i in range(len(notes)):
        d, volume, note = notes[i]
        
        if cum_time == k:
            chromosome_left = [prob_op, prob_mut, tempo, *notes[:i]]
            chromosome_right = [prob_op, prob_mut, tempo, *notes[i:]]
            
            return chromosome_left, chromosome_right
        
        elif cum_time< k < cum_time+d:
            d_left = k - cum_time - 1
            d_right = d - d_left
            
            notes_left = notes[:i+1]
            notes_right = notes[i:]
            
            notes_left[-1] = (d_left, volume, note)
            notes_right[0] = (d_right, volume, note)
            
            chromosome_left = [ prob_op, prob_mut, tempo, *notes_left ]
            chromosome_right = [prob_op, prob_mut, tempo, *notes_right ]
            
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
    prob_op_left, prob_mut_left, tempo_left, notes_left = split_chromosome(left)
    prob_op_right, prob_mut_right, tempo_right, notes_right = split_chromosome(right)
    
    prob_op = ( prob_op_left + prob_op_right ) / 2
    prob_mut = ( prob_mut_left + prob_mut_right) / 2
    tempo = (tempo_left + tempo_right)/2
    notes = notes_left + notes_right
    
    chromosome = [prob_op, prob_mut, tempo, *notes]
    return chromosome



############################################################
############################################################
######                                               #######
######  Functions/Procedures Related to Duplication  #######
######                                               #######
############################################################
############################################################


def duplication(chromosome: Chromosome, *args, **kwargs):
    prob_op, prob_mut, tempo, lines = split_lines(chromosome)

    new_lines = []
    for notes in lines:
        temp_notes = sub_duplication(notes)
        new_lines.append(temp_notes)
    
    chromosome = [ prob_op, prob_mut, tempo, lines ]
    return [chromosome]

def sub_duplication(notes: list[Note], *args, **kwargs) -> list[Chromosome]:
    """
    Given a 'chromosome', randomly selects an interval in the 
    list of notes. That interval is duplicated.
    """    
    i = 0 if len(notes)<=1 else choice( range(len(notes)-1) )
    j = choice( range(1,8+1) )
        
    to_duplicate = notes[i : i+j]
    left = notes[ :i ]
    right = notes[ i+j: ]
    notes = left + to_duplicate + to_duplicate + right
    
    return notes


##########################################################
##########################################################
######                                             #######
######  Functions/Procedures Related to Inversion  #######
######                                             #######
##########################################################
##########################################################

def inversion(chromosome: Chromosome, *args, **kwargs):
    prob_op, prob_mut, tempo, lines = split_lines(chromosome)
    new_lines = []
    for notes in lines:
        new_notes = sub_inversion(notes)
        new_lines.append(notes)

    chromosome = [prob_op, prob_mut,tempo, new_lines]
    return [chromosome]

def sub_inversion(notes: list[Note]) -> list[Chromosome]:
    """
    Given a 'chromosome', randomly selects an interval in the
    list of notes. That interval is reversed.
    """    
    if len(notes)<=1:
        return notes

    i = choice( range(len(notes) - 1) )
    if i == len(notes) -1:
        return notes

    j = choice( range(i+1, len(notes) ) )
                      
    to_reverse = notes[i : j]
    left = notes[:i]
    right = notes[j:]
    
    notes = left + to_reverse[::-1] + right
    
    return notes


#####################################################################
#####################################################################
######                                                        #######
######  Functions/Procedures Related Probabilities Mutations  #######
######                                                        #######
#####################################################################
#####################################################################

def mutate_probabilities(chromosomes: list[Chromosome]) -> list[Chromosome]:
    new_chromosomes = []
    for c in chromosomes:
        prob_op, prob_mut, tempo, lines = split_lines(c)
        op_index = choice(range(4))

        noise = np.random.uniform(low = -0.1, high=0.1, size=1)[0]
        
        prob_op[op_index] = np.abs(prob_op[op_index] + noise)
        prob_op/=prob_op.sum()
        
        if op_index == 0:
            mut_index = choice(range(11))

            noise = noise = np.random.uniform(low = -0.1, high=0.1, size=1)[0]

            prob_mut[mut_index] = np.abs(prob_mut[mut_index] + noise)
            prob_mut/=prob_mut.sum()
        
        new_chromosomes.append([prob_op, prob_mut, tempo, lines])
    return new_chromosomes
        