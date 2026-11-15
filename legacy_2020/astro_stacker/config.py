import os


# Config related stuff. To move later
class Config:
    DATA_DIR = os.path.expanduser('~/pictures')

    COMET_DIR = DATA_DIR + '/2020-07-26-Comet-NEOWISE'

    DARK_DIR = COMET_DIR + '/dark'

    # Different stacks attempted
    STACK_0_DIR = COMET_DIR + '/stack_0'
    STACK_1_DIR = COMET_DIR + '/stack_1'
    STACK_2_DIR = COMET_DIR + '/stack_2'
    STACK_3_DIR = COMET_DIR + '/stack_3'
