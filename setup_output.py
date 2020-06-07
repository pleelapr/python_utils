import os

def setup():
    if not os.path.exists('output'):
        os.makedirs('output')