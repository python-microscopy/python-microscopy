
import numpy as np
from PYME.Analysis.points import cluster_morphology

# cube
x_cube = np.array([1, 1, 1, -1, -1, -1, 1, -1], dtype=float)
y_cube = np.array([1, 1, -1, 1, -1, 1, -1, -1], dtype=float)
z_cube = np.array([1, -1, 1, 1, 1, -1, -1, -1], dtype=float)

def test_anisotropy():
    x = np.arange(10)
    # perfectly anisotropic
    output = cluster_morphology.measure_3d(x, x, x)
    np.testing.assert_almost_equal(output['anisotropy'][0], 1.)

    # isotropic
    output = cluster_morphology.measure_3d(x_cube, y_cube, z_cube)
    np.testing.assert_almost_equal(output['anisotropy'][0], 0.)

def test_principle_axes():
    x = np.arange(10)
    nill = np.zeros_like(x)
    # perfectly anisotropic
    output = cluster_morphology.measure_3d(x, nill, nill)
    np.testing.assert_array_almost_equal(np.array([output['sigma0'][0], output['sigma1'][0], output['sigma2'][0]]),
                                         np.array([np.std(x, ddof=1), 0, 0]))

    np.testing.assert_array_almost_equal(np.array([output['sigma_x'][0], output['sigma_y'][0], output['sigma_z'][0]]),
                                         np.array([np.std(x, ddof=1), 0, 0]))

    np.testing.assert_almost_equal(np.array([1, 0, 0]), output['axis0'][0])

def test_labels_from_image():
    from PYME.IO.image import ImageStack
    from PYME.IO.tabular import DictSource
    im_size = 10
    im = np.zeros((im_size, im_size, im_size), dtype=int)
    im[-3:, -3:, -3:] = 1
    im[:5, :5, :5] = 2

    image_stack = ImageStack(im)
    assert(image_stack.origin == (0, 0, 0) and image_stack.pixelSize == 1)
    image_stack.mdh['voxelsize.x'], image_stack.mdh['voxelsize.y'], image_stack.mdh['voxelsize.z'] = 0.001, 0.001, 0.001

    xx, yy, zz = np.meshgrid(np.arange(im_size), np.arange(im_size), np.arange(im_size))
    points = DictSource({
        'x': xx.ravel(), 'y': yy.ravel(), 'z': zz.ravel()
    })
    points.mdh = image_stack.mdh

    ids, counts_per_label = cluster_morphology.get_labels_from_image(image_stack, points)
    np.testing.assert_array_equal(ids, im.ravel())
    assert counts_per_label[0] == (im == 1).sum()
    assert counts_per_label[1] == (im == 2).sum()

    # now test minimum counts, throwing out the smaller label
    ids, counts_per_label = cluster_morphology.get_labels_from_image(image_stack, points, (im == 1).sum() + 1)
    assert not np.any(ids == 1)
    assert (ids == 2).sum() == (im==2).sum()

