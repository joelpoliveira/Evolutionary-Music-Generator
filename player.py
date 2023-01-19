from pyo import Server, SfPlayer
s = Server().boot()
s.start()
sf = SfPlayer("./src/output/ga-melody-0.wav", speed=1, loop=True).out()