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

    if not origin:
        origin = (0,0,0)
        
    # FIXME - this has a silly memory overhead - write a better version of marching cubes
    #vertices, values = marching_cubes.image_to_vertex_values(data, voxelsize=voxel_size)
    
    #MC = modified_marching_cubes.ModifiedMarchingCubes(isolevel)
    #MC = marching_cubes.MarchingCubes(isolevel)
    #MC.set_vertices(vertices, values)
    #T = triangle_mesh.TriangleMesh.from_np_stl(MC.march())
    
    #MC = marching_cubes.RasterMarchingCubes(data, isolevel, voxelsize=voxel_size)
    MC = modified_marching_cubes.RasterMarchingCubes(data, isolevel, voxelsize=voxel_size)
    T = triangle_mesh.TriangleMesh.from_np_stl(MC.march(), origin=origin)
    
    if remesh:
        T.remesh()
    
    return T

# def triangle_sdf(p, pv):
#     """point plane distance

#     Parameters
#     ----------
#     p : np.array
#         Nx(x,y,z) points
#     pv : np.array
#         NxMx(v0,v1,v2)x(x,y,z) triangle vertex x coordinates
#     """
#     # M = number of faces to average over (see distance_to_mesh)
#     p1p0 = pv[:,:,1] - pv[:,:,0]    # (N x M x (x,y,z))
#     p2p0 = pv[:,:,2] - pv[:,:,0]    # (N x M x (x,y,z))
#     n = np.cross(p1p0,p2p0,axis=2)  # (N x M x (x,y,z))
#     nn = np.linalg.norm(n,axis=2)   # (N x M)
#     nh = n/nn[...,None]             # (N x M x (x,y,z))
#     nh[nn==0] = 0 
#     return (-nh*(p[:,None,:]-pv[:,:,0])).sum(2)  # (N x M)

def triangle_sdf(p, pv):
    """https://iquilezles.org/www/articles/distfunctions/distfunctions.htm

    Parameters
    ----------
    p : np.array
        Nx(x,y,z) points
    pv : np.array
        NxMx(v0,v1,v2)x(x,y,z) triangle vertex x coordinates
    """
    ba = pv[:,:,1] - pv[:,:,0]  
    cb = pv[:,:,2] - pv[:,:,1]
    ac = pv[:,:,0] - pv[:,:,2]
    pa = p[:,None,:] - pv[:,:,0]
    pb = p[:,None,:] - pv[:,:,1]
    pc = p[:,None,:] - pv[:,:,2]
    nor = np.cross(ba,ac,axis=2) 

    s0 = np.sign((np.cross(ba,nor,axis=2)*pa).sum(2))
    s1 = np.sign((np.cross(cb,nor,axis=2)*pb).sum(2))
    s2 = np.sign((np.cross(ac,nor,axis=2)*pc).sum(2))
    cond = (s0+s1+s2)>=2.0

    c0 = ba*np.clip((ba*pa).sum(2)/(ba*ba).sum(2),0.0,1.0)[:,:,None]-pa
    c1 = cb*np.clip((cb*pb).sum(2)/(cb*cb).sum(2),0.0,1.0)[:,:,None]-pb
    c2 = ac*np.clip((ac*pc).sum(2)/(ac*ac).sum(2),0.0,1.0)[:,:,None]-pc

    m = np.minimum(np.minimum((c0*c0).sum(2),(c1*c1).sum(2)),(c2*c2).sum(2))
    norpa = (-nor*pa).sum(2)  # flip the sign so inside the mesh looks like outside
                              # trick 2 from https://iquilezles.org/www/articles/interiordistance/interiordistance.htm
    g = norpa*norpa/(nor*nor).sum(2)

    m[cond] = g[cond]

    s = np.sign(norpa)

    return s*np.sqrt(m)

def sdf_min(sdf, smooth=True, k=0.1):
    # TODO: Default value of k=0.1 is very much arbitrary. Is there a better choice?
    # Means the function is smoothed by ~1/10 nm
    if smooth:
        # Exponentially smoothed minimum
        # https://iquilezles.org/www/articles/smin/smin.htm
        val = -np.log2(np.nansum(np.exp2(-k*sdf), axis=1))/k
        val[np.isnan(val)] = 0  # log2 of 0 or a negative number, shouldn't happen
        return val
    else:
        return np.min(sdf, axis=1)

def distance_to_mesh(points, surf, smooth=False, smooth_k=0.1, tree=None):
    """
    Calculate the distance to a mesh from points in a tabular dataset 

    Parameters
    ----------
        points : np.array
            3D point cloud to fit (nm).
        surf : PYME.experimental._triangle_mesh
            Isosurface
        smooth : bool
            Smooth distance to mesh?
        smooth_k : float
            Smoothing constant, by default 0.1 = smoothed by 1/10 nm
        tree : scipy.spatial.cKDTree
            Optionally pass a tree of face centers to use if calling
            this function multiple times.
    """

    if tree is None:
        import scipy.spatial

        # Create a list of face centroids for search
        face_centers = surf._vertices['position'][surf.faces].mean(1)

        # Construct a kdtree over the face centers
        tree = scipy.spatial.cKDTree(face_centers)

    # Get M closet face centroids for each point
    M = 5
    _, _faces = tree.query(points, k=M, workers=-1)
    
    # Get position representation
    _v = surf.faces[_faces]
    v = surf._vertices['position'][_v]  # (n_points, M, (v0,v1,v2), (x,y,z))

    # the negative reverses the norpa sign flip
    # trick 2 from https://iquilezles.org/www/articles/interiordistance/interiordistance.htm
    return -sdf_min(triangle_sdf(points, v), smooth=smooth, k=smooth_k)
