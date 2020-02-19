import numpy as np


def load_stl_binary(fn):
    """
    Load triangles from binary STL file.

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
        n_triangles = np.fromfile(f, 'int32', 1).squeeze()
        triangles = np.fromfile(f, [('normal', '3f4'), ('vertex0', '3f4'), ('vertex1', '3f4'), ('vertex2', '3f4'), ('attrib', 'u2')], n_triangles)

        return triangles


def save_stl_binary(fn, data):
    """
    Save list of triangles to binary STL file.

    Parameters
    ----------
    fn : string
        File name to write
    data : np.array
        A list of triangles in the same format as returned by load_stl_binary

    Returns
    -------
    None
    """

    with open(fn, 'wb') as f:
        f.write(np.zeros(80,dtype='uint8'))
        f.write(np.uint32(data.shape[0]))
        data.tofile(f)
