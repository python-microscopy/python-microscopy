import numpy

# this should serve as a drop in replacement for numpy.load calls that need
# some backwards compatibility
def np_load_legacy(f):
    try:
        ret = numpy.load(f, allow_pickle=True)
    except OSError:
        # this one based on tips from https://rebeccabilbro.github.io/convert-py2-pickles-to-py3/
        ret = numpy.load(f, allow_pickle=True, encoding="latin1")
    return ret
