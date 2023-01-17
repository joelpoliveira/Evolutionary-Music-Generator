import os

from scipy.io import wavfile
from midi2audio import FluidSynth
from midiutil import MIDIFile
from io import BytesIO
from util.features import get_feature_vector

FONT = "piano_chords.sf2"
MIDI_FILE = "major-scale.midi"
WAV_FILE = "major-scale.wav"

degrees  = [60, 62, 64, 65, 67, 69, 71, 72] # MIDI note number
degrees = [4, 6, 8, 9, 11, 13, 15, 16]
track    = 0
channel  = 0
time     = 0   # In beats
duration = 1   # In beats
tempo    = 100  # In BPM
volume   = 100 # 0-127, as per the MIDI standard


MyMIDI = MIDIFile(1) # One track, defaults to format 1 (tempo track
                     # automatically created)
MyMIDI.addTempo(track,time, tempo)

for pitch in degrees:
    MyMIDI.addNote(track, channel, pitch, time, duration, volume)
    time = time + 1

with open(f"./input/{MIDI_FILE}", "wb") as output_file:
    MyMIDI.writeFile(output_file)

song = BytesIO()
wav_song = BytesIO( )

MyMIDI.writeFile(song)
FluidSynth(
        sound_font=f"./soundfonts/{FONT}"
    ).midi_to_audio(f"./input/{MIDI_FILE}", f"./output/{WAV_FILE}")

fs, song = wavfile.read(f"./output/{WAV_FILE}")
#print(song)
#sd.play(song, fs)
#sd.wait()
# sd.play()
#print(get_feature_vector(song))


if __name__=="__main__":
    pass