
from PYME.IO import tabular
from PYME.IO.MetaDataHandler import NestedClassMDHandler
from PYME.recipes import measurement, base, tablefilters
import numpy as np
from scipy.spatial import KDTree

import pytest

def test_IdentifyOverlappingROIs():
    mdh = NestedClassMDHandler()
    mdh['voxelsize.x'] = 0.115
    roi_size = 256
    roi_size_um = roi_size * mdh['voxelsize.x']
    max_distance = np.sqrt(2) * roi_size_um
    points = tabular.RandomSource(100 * roi_size_um, 100 * roi_size_um, int(1e3))
    points = tabular.MappingFilter(points, **{'x_um': 'x', 'y_um': 'y'})  # pretend we defined points in um
    points.mdh = mdh

    recipe = base.ModuleCollection()
    recipe.add_module(measurement.IdentifyOverlappingROIs(roi_size_pixels=roi_size, output='mapped'))
    recipe.add_module(tablefilters.FilterTable(inputName='mapped', filters={'rejected': [-0.5, 0.5]},
                                               outputName='output'))
    recipe.namespace['input'] = points
    filtered = recipe.execute()

    positions = np.stack([filtered['x'], filtered['y']], axis=1)

    kdt = KDTree(positions)
    distances, indices = kdt.query(positions, k=2, p=2)
    assert (distances[:, 1] > max_distance).all()

def test_TravelingSalesperson():
    r = 10
    theta = np.linspace(0, 2* np.pi, 5)
    dt = theta[1] - theta[0]
    x, y = r * np.cos(theta), r * np.sin(theta)
    x = np.concatenate([x, r * np.cos(theta + 0.5 * dt)])
    y = np.concatenate([y, r * np.sin(theta + 0.5 * dt)])

    points = tabular.DictSource({'x_um': np.concatenate([x, 1.1 * r * np.cos(theta)]),
                                    'y_um': np.concatenate([y, 1.1 * r * np.sin(theta)])})

    recipe = base.ModuleCollection()
    recipe.add_module(measurement.TravelingSalesperson(output='output', epsilon=0.001))
    recipe.namespace['input'] = points

    ordered = recipe.execute()

    # should be not too much more than the rough circumference.
    assert ordered.mdh['TravelingSalesperson.Distance'] < 1.25 * (2 * np.pi * r)

@pytest.mark.xfail(reason='Requires sklearn, which is not installed for testing to ease dependency management')
def test_ChunkedTravelingSalesman():
    n = 500
    x = np.random.rand(n) * 4e3
    y = np.random.rand(n) * 4e3

    points = tabular.DictSource({'x_um': x, 'y_um': y})

    recipe = base.ModuleCollection()
    recipe.add_module(measurement.ChunkedTravelingSalesperson(output='output', epsilon=0.001,
                                                              points_per_chunk=50))
    recipe.namespace['input'] = points

    ordered = recipe.execute()
    assert ordered.mdh['TravelingSalesperson.Distance'] < ordered.mdh['TravelingSalesperson.OriginalDistance']


if __name__ == '__main__':
    import time
    n = 10000
    x = np.random.rand(n) * 4e3
    y = np.random.rand(n) * 4e3

    points = tabular.DictSource({'x_um': x, 'y_um': y})

    recipe = base.ModuleCollection()
    recipe.add_module(measurement.ChunkedTravelingSalesperson(output='output', epsilon=0.001,
                                                              points_per_chunk=500))
    recipe.namespace['input'] = points

    t = time.time()
    ordered = recipe.execute()
    print('n_points: %d, runtime: %f' % (n, time.time() - t))
    print('og distance: %f, distance: %f' % (ordered.mdh['TravelingSalesperson.OriginalDistance'], ordered.mdh['TravelingSalesperson.Distance']))
