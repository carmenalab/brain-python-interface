import pygame
import os

audio_path = os.path.dirname(__file__)

class AudioPlayer():

    def __init__(self, filename='click.wav'):
        if not pygame.mixer.get_init():
            pygame.mixer.pre_init(44100, -16, 2, 2048)
            pygame.mixer.init()
        self.effect = pygame.mixer.Sound(os.path.join(audio_path, filename))

    def play(self):
        self.effect.play()
