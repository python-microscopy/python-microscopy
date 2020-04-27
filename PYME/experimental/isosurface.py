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

def dot(v, w):
    return (v*w).sum()

def dot2(v):
    return (v*v).sum()

def clamp(v, lo, hi):
    if v < lo:
        return lo
    if hi < v:
        return hi
    return v

def clamp2(v, lo, hi):
    out = v
    out[v < lo] = lo
    out[hi < v] = hi
    return out

def fast_3x3_cross(a,b):
    x = a[1]*b[2] - a[2]*b[1]
    y = a[2]*b[0] - a[0]*b[2]
    z = a[0]*b[1] - a[1]*b[0]

    vec = np.array([x,y,z])
    return vec

def sign(x):
    if x > 0:
        return 1
    return -1

def triangle_sdf(p, p0, p1, p2):
    # Calculate vector distances
    p1p0 = p1 - p0
    p2p1 = p2 - p1
    p0p2 = p0 - p2
    pp0 = p - p0
    pp1 = p - p1
    pp2 = p - p2
    n = fast_3x3_cross(p1p0, p0p2)
    
    s1 = sign(dot(fast_3x3_cross(p1p0,n),pp0))
    s2 = sign(dot(fast_3x3_cross(p2p1,n),pp1))
    s3 = sign(dot(fast_3x3_cross(p0p2,n),pp2))
    if (s1+s2+s3) < 2:
        f = min(min(dot2(p1p0*clamp(dot(p2p1,pp0)/dot2(p1p0),0.0,1.0)-pp0),
                    dot2(p2p1*clamp(dot(p2p1,pp1)/dot2(p2p1),0.0,1.0)-pp1)),
                    dot2(p0p2*clamp(dot(p0p2,pp2)/dot2(p0p2),0.0,1.0)-pp2))
    else:
        f = dot(n,pp0)*dot(n,pp0)/dot2(n)
    
    return sign(dot(pp0,n))*(f**0.5)

def triangle_sdf2(p, pv):
    p1p0 = pv[:,1] - pv[:,0]
    p2p1 = pv[:,2] - pv[:,1]
    p0p2 = pv[:,0] - pv[:,2]
    pp0 = p - pv[:,0]
    pp1 = p - pv[:,1]
    pp2 = p - pv[:,2]
    n = np.cross(p1p0, p0p2, axis=1)

    s1 = np.sign((np.cross(p1p0,n,axis=1)*pp0).sum(1))
    s2 = np.sign((np.cross(p2p1,n,axis=1)*pp1).sum(1))
    s3 = np.sign((np.cross(p0p2,n,axis=1)*pp2).sum(1))

    f = (n*pp0).sum(1)*(n*pp0).sum(1)/(n*n).sum(1)
    sign_mask = (s1+s2+s3) < 2
    f[sign_mask] = np.minimum(np.minimum(
                    ((p1p0*clamp2((p2p1*pp0).sum(1)/(p1p0*p1p0).sum(1),0.0,1.0)[:,None]-pp0)**2).sum(1),
                    ((p2p1*clamp2((p2p1*pp1).sum(1)/(p2p1*p2p1).sum(1),0.0,1.0)[:,None]-pp1)**2).sum(1)),
                    ((p0p2*clamp2((p0p2*pp2).sum(1)/(p0p2*p0p2).sum(1),0.0,1.0)[:,None]-pp2)**2).sum(1))[sign_mask]

    return np.sign((pp0*n).sum(1))*np.sqrt(f)

def distance_to_isosurface(points, surf):
    """
    Calculate the distance from points in a tabular dataset 

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

    # Vector of distances
    d = np.zeros(v.shape[0])

    # Calculate planar distances from each point to its candidate triangles
    # and choose the minimum absolute value
    for _i in range(v.shape[0]):
        # d_min = 1e15
        # for _j in range(N):
        #     d_tmp = triangle_sdf(points[_i], *v[_i,_j])
        #     if abs(d_tmp) < abs(d_min):
        #         d_min = d_tmp
        # d[_i] = d_min
        d_tmp = triangle_sdf2(points[_i], v[_i,...])
        d[_i] = d_tmp[np.argmin(np.abs(d_tmp))]

    return d
