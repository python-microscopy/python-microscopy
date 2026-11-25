import matplotlib
matplotlib.use('wxagg')

import numpy as np
import matplotlib.pyplot as plt

from PYME.recipes import Recipe
from PYME.recipes import pointcloud
from PYME.IO import tabular

def test_iterative_closest_point():

    from PYME.simulation import locify
    def round_box(p, w, r):
        w = np.array(w)
        q = np.abs(p) - w[:,None]
        return np.linalg.norm(np.maximum(q,0.0),axis=0) + np.minimum(np.maximum(q[0,:],np.maximum(q[1,:],q[2,:])),0.0) - r
    
    shift = [0.02,0,0]
    rot = np.pi/12
    cube0 = locify.points_from_sdf(lambda x: round_box(x, [0.5,0.5,0.5], 0), r_max=1, centre=(0,0,0), dx_min=0.2, p=1.0)
    cube1 = cube0.copy()
    cube1 += np.array(shift)[:,None]
    cube1 = np.dot(np.array([[np.cos(rot), 0, np.sin(rot)], [0, 1, 0], [-np.sin(rot), 0, np.cos(rot)]]), cube1)

    weights_reference = np.random.rand(*cube0.shape)
    weights_target = np.random.rand(*cube1.shape)


    cube0_ds = tabular.DictSource({"x": cube0[0,:],
                                   "y": cube0[1,:],
                                   "z": cube0[2,:],
                                   "error_x": weights_reference[0,:],
                                   "error_y": weights_reference[1,:],
                                   "error_z": weights_reference[2,:]})
    
    cube1_ds = tabular.DictSource({"x": cube1[0,:],
                                   "y": cube1[1,:],
                                   "z": cube1[2,:],
                                   "error_x": weights_target[0,:],
                                   "error_y": weights_target[1,:],
                                   "error_z": weights_target[2,:]})
    

    recipe = Recipe()
    recipe.namespace['reference'] = cube0_ds
    recipe.namespace['to_register'] = cube1_ds

    icp = pointcloud.IterativeClosestPoint(reference='reference', 
                                           to_register='to_register', 
                                           output='output', 
                                           max_iters=5,
                                           max_points=50,
                                           sigma_x="nerf", 
                                           sigma_y="nerf", 
                                           sigma_z="nerf")
    recipe.add_module(icp)

    res = recipe.execute()

    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(cube0[0,...], cube0[1,...],cube0[2,...])
    # ax.scatter(cube1[0,...], cube1[1,...],cube1[2,...])
    # ax.scatter(res['x'], res['y'], res['z'])
    # # ax.scatter(res['xp'], res['yp'], res['zp'])
    # ax.set_xlim([-1,1])
    # ax.set_ylim([-1,1])
    # ax.set_zlim([-1,1])

    # plt.show()

    np.testing.assert_allclose(cube0[0,:], res['x'])
    np.testing.assert_allclose(cube0[1,:], res['y'])
    np.testing.assert_allclose(cube0[2,:], res['z'])

    icp2 = pointcloud.IterativeClosestPoint(reference='reference', 
                                           to_register='to_register', 
                                           output='output', 
                                           max_iters=5,
                                           max_points=50)
    recipe.add_module(icp2)

    res2 = recipe.execute()

    np.testing.assert_array_less(np.abs(cube0[0,:]- res2['x']), 1/weights_reference[0,:])
    np.testing.assert_array_less(np.abs(cube0[1,:], res2['y']), 1/weights_reference[1,:])
    np.testing.assert_array_less(np.abs(cube0[2,:], res2['z']), 1/weights_reference[2,:])

