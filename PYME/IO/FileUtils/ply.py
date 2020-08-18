import numpy as np

def load_ply(fn):
    """
    Load PLY from file.
    """

    with open(fn, 'rb+') as f:
        line = ''
        order = ''
        file_format = None
        colors = False
        n_vertices = 0
        n_faces = 0
        while not line.startswith('end_header'):
            line = f.readline().decode('ascii')
            if line.startswith('format'):
                file_format = line.split(' ')[1]
            if line.startswith('property uchar red'):
                # We assume this means a full RGB profile
                colors = True
            if line.startswith('element'):
                split_line = line.split(' ')
                if split_line[1] == 'vertex':
                    n_vertices = int(split_line[2])
                elif split_line[1] == 'face':
                    n_faces = int(split_line[2])
        if file_format == 'binary_little_endian':
            order = '<'
        if file_format == 'binary_big_endian':
            order = '>'

        if file_format == 'ascii':
            vertices = np.zeros((n_vertices, 3), dtype=np.float32)
            if colors:
                color = np.zeros((n_vertices, 3), dtype=np.ubyte)
            faces = np.zeros((n_faces, 3), dtype=np.int32)
            for l in np.arange(n_vertices):
                line = f.readline().decode('ascii').strip().split(' ')
                vertices[l, 0] = np.float32(line[0])
                vertices[l, 1] = np.float32(line[1])
                vertices[l, 2] = np.float32(line[2])
                if colors:
                    color[l, 0] = np.ubyte(line[3])
                    color[l, 1] = np.ubyte(line[4])
                    color[l, 2] = np.ubyte(line[5])
            for l in np.arange(n_faces):
                line = f.readline().decode('ascii').strip().split(' ')
                n_elements = np.int32(line[0])
                faces[l, 0] = np.int32(line[1])
                faces[l, 1] = np.int32(line[2])
                faces[l, 2] = np.int32(line[3])
            if colors:
                return vertices, faces, color
            else:
                return vertices, faces, np.zeros_like(vertices)
        else:
            if colors:
                vc = np.fromfile(f, [('vertices', order+'3f4'), ('colors', '3u1')], n_vertices)
            else:
                vc = np.fromfile(f, order+'3f4', n_vertices)
            faces = np.fromfile(f, [('n_elements', 'u1'), ('elements', order+'3i4')] , n_faces)
            if colors:
                return vc['vertices'], faces['elements'], vc['colors']
            else:
                 return vc, faces['elements'], np.zeros_like(vc)
        
def save_ply(fn, vertices, faces, colors=None, file_format='binary'):
    """
    Save list of triangles to PLY file.

    Parameters
    ----------
    fn : string
        File name to write
    vertices : np.array or list
        An n_vertices x 3 array or list of floats.
    faces : np.array or list
        An n_faces x m (usually m=3) array of indexes to vertices. Faces can
        describe any polygon with 3 or more sides.
    colors : np.array
        A n_vertices x 3 array of RGB colors.
    file_format : str
        Write PLY as ascii or binary
    
    Returns
    -------
    None
    """

    file_formats = ['ascii', 'binary']
    if file_format not in file_formats:
        print('Weird file format. Defaulting to ascii.')
        file_format = 'ascii'

    mode = 'w'
    if file_format == 'binary':
        mode = 'wb'
        import sys
        if sys.byteorder == 'little':
            file_format = 'binary_little_endian'
            order = '<'
        else:
            file_format = 'binary_big_endian'
            order = '>'

    with open(fn, mode) as f:
        h_str = ''
        h_str += 'ply\nformat {} 1.0\n'.format(file_format)
        h_str += 'element vertex {}\n'.format(len(vertices))
        h_str += 'property float x\nproperty float y\nproperty float z\n'
        if colors is not None:
            h_str += 'property uchar red\nproperty uchar green\nproperty uchar blue\n'
        h_str += 'element face {}\n'.format(len(faces))
        h_str += 'property list uchar int vertex_index\n'
        h_str += 'end_header\n'
        if file_format == 'ascii':
            f.write(h_str)
            if colors is not None:
                for _iv in np.arange(len(vertices)):
                    f.write('{} {} {} {} {} {}\n'.format(vertices[_iv,0], 
                                                        vertices[_iv,1], 
                                                        vertices[_iv,2], 
                                                        colors[_iv, 0], 
                                                        colors[_iv, 1], 
                                                        colors[_iv, 2]))
            else:
                for _iv in np.arange(len(vertices)):
                    f.write('{} {} {}\n'.format(vertices[_iv,0],
                                                vertices[_iv,1], 
                                                vertices[_iv,2]))
            for _if in np.arange(len(faces)):
                curr_faces = faces[_if]
                f.write(str(curr_faces.size) + ' ' + ' '.join(str(x) for x in curr_faces) + '\n')
        else:
            f.write(h_str.encode('ascii'))
            if colors is not None:
                for _iv in np.arange(len(vertices)):
                    f.write(vertices[_iv].astype(order + 'f4'))
                    f.write(colors[_iv].astype(order + 'u1'))
            else:
                for _iv in np.arange(len(vertices)):
                    f.write(vertices[_iv].astype(order + 'f4'))
            for _if in np.arange(len(faces)):
                curr_faces = faces[_if]
                f.write(np.ubyte(curr_faces.size))
                f.write(curr_faces.astype(order + 'i4'))
