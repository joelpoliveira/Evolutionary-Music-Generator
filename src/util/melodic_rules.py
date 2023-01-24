import numpy as np
from itertools import combinations
from util.ga import split_chromosome

MIN_NOTE = 36
MAX_NOTE = 96

def get_score(chromosome):
    _,_,_,notes = split_chromosome(chromosome)
    
    f_w = {
        opening_chord: 0.25, #1
        closing_chord: 0.25, #1
        final_cadence: 1,
        parts_in_range: 1,
        leap_height: 1,
        leap_resolution: 1,
        voice_range: 1,
        repeating_pitch: 1,
        half_stepwise: 1,
        is_sufficient: 1,
        is_monotonous: 1,
        note_repetition: 1,
        duration_balance: 1
    }
    total = sum(f_w.values())
    score = sum( f(notes) * w for f,w in f_w.items() ) / total
    return score


def is_sufficient(notes):
    MIN_THRESH = 5
    MAX_THRESH = 50

    score=0
    if MIN_THRESH<len(notes)<MAX_THRESH:
        score+=1

    return score/len(notes)


def is_monotonous(notes):
    score = 0
    notes_in = set()
    notes = list(map(lambda note: note[-1] , notes))

    for note in notes:
        notes_in|={note}

    max_note = np.max(notes)
    min_note = np.min(notes)
    
    pitch_range = (max_note-min_note)/(MAX_NOTE-MIN_NOTE)
    variety = len(notes_in) / len(notes)
    
    score+= min(pitch_range, variety)
    return score


def is_tonic(notes, i):
    _, _, note = notes[i]

    note_i = note%12
    return note_i in (0, 4, 7) 


def opening_chord(notes):
    score = 0
    if is_tonic(notes, 0):
        score += 1
    return score


def closing_chord(notes):
    score = 0
    if is_tonic(notes, -1):
        score += 1
    return score


def diatonic_lower_step_size(note) -> int:
    if note%12 in (0, 5):
        return 1
    return 2


def diatonic_upper_step_size(note) -> int:
    if note%12 in (0, 5):
        return 2
    return 1


def authentic_cadence(note_a, note_b):
    root_a = note_a - note_a%12
    root_b = note_b - note_b%12

    if ( note_a%12 in (7,6) ) and (note_b%12==0) and (root_a==root_b):
        return True
    return False


def is_diatonic_step(note_a, note_b):
    if note_a>note_b:
        if note_a - note_b == diatonic_lower_step_size(note_a):
            return True
    elif note_a<note_b:
        if note_b - note_a == diatonic_upper_step_size(note_a):
            return True
    return False


def final_cadence(notes):
    score = 0
    try:
        last_note = notes[-1][-1]
        bef_last_note = notes[-2][-1]
    except Exception:
        return 0
    
    if authentic_cadence(last_note, bef_last_note):
        score+=1

    return score


def parts_in_range(notes):
    score = 0

    pitches = list(map(lambda note: note[-1], notes))

    max_pitch = np.max(pitches)
    min_pitch = np.min(pitches)
        
    if (max_pitch - min_pitch) <= 23 and (max_pitch != min_pitch):
        score+= 1

    return score


def leap_height(notes):
    score = 0

    n = len(notes)-1
    if n!=0:
        temp_score=0
        for i in range(n):
            if abs(notes[i][-1] - notes[i+1][-1]) <= 14:
                temp_score+=1
        score += (temp_score / n)
    
    return score


def leap_resolution(notes):
    score = 0

    counter = 0
    n = len(notes) - 2
    if n>1:
        for i in range(n):
            a = notes[i][-1]
            b = notes[i+1][-1]
            c = notes[i+2][-1]

            if abs(a - b)>8:
                counter+=1
                if ( (a < b) and 
                ( b - diatonic_lower_step_size(b) ) == c):
                    score+=1
                elif ( (a> b) and
                ( b + diatonic_upper_step_size(b) ) == c):
                    score+=1
        if counter!=0:
            score /= counter
    return score


def get_time_left(notes, t, i):
    if i>=len(notes):
        return [], i

    used = []
    d, v, note = notes[i]

    if d+t==1:
        used.append(note)
        return used, i+1
    elif d+t>1:
        d = d - (1-t)
        notes[i] = (d, v, note)
        used.append(note)
        return used, i
    elif d+t<1:
        used.append(note)
        also_used, new_i = get_time_left(notes, t+d, i+1)
        return used+also_used, new_i


def get_next_second(notes, i):
    t = 0
    used = []
    while t<1:
        if i>=len(notes): break
        d, v, note = notes[i]

        if t + d ==1:
            used.append(note)
            return (1, v, np.mean(note)), i+1
        if t + d > 1:
            d = d - (1-t)
            notes[i] = (d, v, note)
            return (1, v, np.mean(note)), i
        if t + d < 1:
            used.append(note)
            t+=d
            also_used, i = get_time_left(notes, t, i+1) 
            used+=also_used
    return (t, v, np.mean(used)), i
       

def notes_uniform(notes):
    notes = notes.copy()
    uniform = []
    i = 0
    while i < len(notes):
        note, i = get_next_second(notes, i)
        uniform.append(note)
    return uniform
                

def voice_range(notes):
    score=0
    n = len(notes)
    for note in notes:
        if MIN_NOTE<=note[-1]<=MAX_NOTE:
            score+=1
    if n!=0:
        score /= n
    return score


def repeating_pitch(notes):
    score = 1

    last_note = None
    n = len(notes)-4

    if n<=0:
        return 0

    counter=0
    for note in notes:
        _, _, pitch = note
        if last_note == pitch:
            counter+=1
        else:
            counter=0

        if counter==4:
            score -= 1/n

        last_note = pitch

    return score
            

def half_stepwise(notes):
    score=0
    n = len(notes) - 1

    stepwise_count=0
    not_stepwise_count=0
    if n>1:
        for i in range(n):
            a = notes[i][-1]
            b = notes[i+1][-1]
            if is_diatonic_step(a, b):
                stepwise_count+=1
            else:
                not_stepwise_count+=1

        score += (stepwise_count)/(stepwise_count+not_stepwise_count)
    return abs(0.5 - score)



#
#
# Rythm Score
#
#

def get_rythm_score(chromosome):
    _,_,_, notes = split_chromosome(chromosome)
    score = note_repetition(notes) + rythm_variation(notes)

    return score/2


def note_repetition(notes):
    score = 1

    n = len(notes)-4
    if n<=0:
        return 0

    counter=0
    last_duration = None
    
    for note in notes:
        d, _, _ = note
        if last_duration == d:
            counter+=1
        else:
            counter=0

        if counter==4:
            score-= 1/n

        last_duration = d

    return score


def duration_balance(notes):
    REFERENCE = 0.3

    short_note=0
    long_note=0

    for note in notes:
        d, _, _ = note
        if d>2:
            long_note+=1
        else:
            short_note+=1
    
    long_ratio = long_note/(long_note + short_note)

    return 1 - abs(long_ratio - REFERENCE)