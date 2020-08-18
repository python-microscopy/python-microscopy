import os

def load_template(filename):
    dir = os.path.join(os.path.dirname(__file__), 'templates')
    
    with open(os.path.join(dir, filename)) as f:
        return f.read()