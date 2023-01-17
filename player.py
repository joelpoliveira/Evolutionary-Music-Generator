from pyo import *
s = Server().boot()
s.start()
sf = SfPlayer("major-scale.mid", speed=1, loop=True).out()