import os

def setup_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)