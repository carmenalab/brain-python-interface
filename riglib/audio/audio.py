import pygame
import os

class AudioPlayer():

    def __init__(self, filename='click.wav'):
        pygame.mixer.pre_init(44100, -16, 2, 2048)
        pygame.mixer.init()
        self.effect = pygame.mixer.Sound(os.path.join('../riglib/audio/', filename))

    def play(self):
        self.effect.play()
