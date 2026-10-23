# This files handles log messages with support for different log-modes.
# Author: Stefan Kahl, 2018, Chemnitz University of Technology

import config as cfg

def show(s, new_line=False):

    if isinstance(s, (list, tuple)):
        print(*s, end='\n' if new_line else ' ')
    else:
        print(s, end='\n' if new_line else ' ')

def i(s, new_line=True):

    if cfg.LOG_MODE in ['all', 'info']:
        show(s, new_line)

def p(s, new_line=True):

    if cfg.LOG_MODE in ['all', 'progress']:
        show(s, new_line)

def e(s, new_line=True):

    if cfg.LOG_MODE in ['all', 'error']:
        show(s, new_line)

def r(s, new_line=True):

    if cfg.LOG_MODE in ['all', 'result']:
        show(s, new_line)
