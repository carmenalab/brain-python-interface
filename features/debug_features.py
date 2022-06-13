import cProfile
import pstats

class Profiler():
    
    def run(self):
        pr = cProfile.Profile()
        pr.enable()
        super().run()
        pr.disable()
        with open('profile.csv', 'w') as f:
            ps = pstats.Stats(pr, stream=f).sort_stats('time')
            ps.print_stats()