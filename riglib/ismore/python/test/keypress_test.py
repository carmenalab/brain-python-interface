# import pygame
import time

# pygame.init()
# screen = pygame.display.set_mode((100,100))

# finished = False

# while not finished:
#     print 'hello'
#     # keys = pygame.key.get_pressed()

#     # if keys[pygame.K_RIGHT]: 
#     #     print 'right'
#     # if keys[pygame.K_LEFT]: 
#     #     print 'left'
#     # if keys[pygame.K_DOWN]: 
#     #     print 'down'
#     # if keys[pygame.K_UP]: 
#     #     print 'up'
#     #     break
#     time.sleep(1)

#     events = pygame.event.get()
#     for event in events:
#         if event.type == pygame.KEYDOWN:
#             if event.key == pygame.K_LEFT:
#                 print 'left'
#             if event.key == pygame.K_RIGHT:
#                 print 'right'
#             if event.key == pygame.K_DOWN:
#                 finished = True

import pygame
import numpy as np

pygame.init()

screen = pygame.display.set_mode((200, 200))
run = True
# pos = (100, 100)
# clock = pygame.time.Clock()

# key bindings
# move_map = {pygame.K_LEFT: np.array([-1, 0]),
#             pygame.K_RIGHT: np.array([1, 0]),
#             pygame.K_UP: np.array([0, -1]),
#             pygame.K_DOWN: np.array([0, 1])}

while run:
    # determine movement vector
    pressed = pygame.key.get_pressed()
    print "pressed", pressed
    # move_vector = np.array([0, 0])
    # for m in (move_map[key] for key in move_map if pressed[key]):
    #     move_vector += m

    # print move_vector

    time.sleep(0.1)
    pygame.event.pump()

    # clock.tick(0.1)