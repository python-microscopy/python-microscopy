import os

def load_template(filename):
    dir = os.path.dirname(__file__)
    
    with open(os.path.join(dir, filename)) as f:
        return f.read()