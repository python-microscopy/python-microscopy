import numpy as np

def load_stl_binary(fn):
    """

    Parameters
    ----------
    fn : string
        File name to read in

    Returns
    -------
    A set of triangles.
    """
    with open(fn, 'rb') as f:
        f.seek(80)
        n_triangles = np.fromfile(f, 'int32', 1)
        triangles = np.fromfile(f, [('normal', '3f4'), ('vertex0', '3f4'), ('vertex1', '3f4'), ('vertex2', '3f4'), ('attrib', 'u2')], n_triangles)

        return triangles

class StlFileSource