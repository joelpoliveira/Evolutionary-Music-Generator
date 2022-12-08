import sounddevice as sd
from scipy.io import wavfile
import fluidsynth
from midi2audio import FluidSynth
from midiutil import MIDIFile
from io import BytesIO

FONT = "8bit.sf2"
degrees  = [60, 62, 64, 65, 67, 69, 71, 72] # MIDI note number
track    = 0
channel  = 0
time     = 0   # In beats
duration = 1   # In beats
tempo    = 60  # In BPM
volume   = 100 # 0-127, as per the MIDI standard


MyMIDI = MIDIFile(1) # One track, defaults to format 1 (tempo track
                     # automatically created)
MyMIDI.addTempo(track,time, tempo)

for pitch in degrees:
    MyMIDI.addNote(track, channel, pitch, time, duration, volume)
    time = time + 1

song = BytesIO()
wav_song = BytesIO( )

MyMIDI.writeFile(song)
FluidSynth(sound_font=F"./soundfonts/{FONT}").midi_to_audio("./major-scale.mid", "./major-scale.wav")

fs, song = wavfile.read("./major-scale.wav")
print(song)
sd.play(song, fs)
sd.wait()
# sd.play()
#play(song, samplerate=44100)

# with open("major-scale.mid", "wb") as output_file:
#     MyMIDI.writeFile(output_file)