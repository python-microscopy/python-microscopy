import numpy as np

from . import modified_marching_cubes
from . import marching_cubes

def isosurface(data, isolevel, voxel_size=None, origin=None, remesh=False):
    """
    Wrapper function around modified marching cubes to generate an isosurface mesh from an image volume
    """
    from PYME.experimental import _triangle_mesh as triangle_mesh
    if not voxel_size:
        voxel_size=1.0
        
    # FIXME - this has a silly memory overhead - write a better version of marching cubes
    #vertices, values = marching_cubes.image_to_vertex_values(data, voxelsize=voxel_size)
    
    #MC = modified_marching_cubes.ModifiedMarchingCubes(isolevel)
    #MC = marching_cubes.MarchingCubes(isolevel)
    #MC.set_vertices(vertices, values)
    #T = triangle_mesh.TriangleMesh.from_np_stl(MC.march())
    
    #MC = marching_cubes.RasterMarchingCubes(data, isolevel, voxelsize=voxel_size)
    MC = modified_marching_cubes.RasterMarchingCubes(data, isolevel, voxelsize=voxel_size)
    T = triangle_mesh.TriangleMesh.from_np_stl(MC.march())
    
    if remesh:
        T.remesh()
    
    return T

def triangle_sdf(p, pv):
    """point plane distance

    Parameters
    ----------
    p : np.array
        Nx3 points
    pv : np.array
        3x3 triangle vertex x coordinates
    """
    p1p0 = pv[:,:,1] - pv[:,:,0]
    p2p0 = pv[:,:,2] - pv[:,:,0]
    n = np.cross(p1p0,p2p0,axis=2)
    nn = np.linalg.norm(n,axis=2)
    nh = n/nn[...,None]
    nh[nn==0] = 0
    return (-nh*(p[:,None,:]-pv[:,:,0])).sum(1)

def distance_to_mesh(points, surf):
    """
    Calculate the distance to a mesh from points in a tabular dataset 

    Parameters
    ----------
        points : np.array
            3D point cloud to fit (nm).
        surf : PYME.experimental._triangle_mesh
            Isosurface
    """

    import scipy.spatial

    # Create a list of face centroids for search
    face_centers = surf._vertices['position'][surf.faces].mean(1)

    # Construct a kdtree over the face centers
    tree = scipy.spatial.cKDTree(face_centers)

    # Get N closet faces for each point
    N = 5
    _, _faces = tree.query(points, k=N)
    
    # Get position representation
    _v = surf.faces[_faces]
    v = surf._vertices['position'][_v]  # (points, N, (v0,v1,v2), (x,y,z))

    # Intersect (np.max) SDFs of nearest triangles
    return np.max(triangle_sdf(points, v),axis=1)
