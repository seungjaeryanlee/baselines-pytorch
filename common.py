import os
import sys
import platform

import numpy as np


OPERATING_SYSTEM = platform.system()

if OPERATING_SYSTEM == 'Windows':
    def clear_terminal():
        os.system('cls')
elif OPERATING_SYSTEM in ['Darwin', 'Linux']:
    def clear_terminal():
        os.system('clear')
else:
    print('Unknown OS: ', OPERATING_SYSTEM)
    sys.exit(1)

def get_running_mean(arr, window=30):
    return np.convolve(arr, np.ones((window,)) / window, mode='valid')

def get_q_map(env, agent):
    q_map = ''
    for state in range(env.observation_space.n):
        amax = np.argmax(agent.q_table[state])
        if amax == 0:
            q_map += ' ← '
        elif amax == 1:
            q_map += ' ↓ '
        elif amax == 2:
            q_map += ' → '
        elif amax == 3:
            q_map += ' ↑ '
        if state % 8 == 7:
            q_map += '\n'

    return q_map
