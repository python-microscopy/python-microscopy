
from PYME.recipes import Recipe
from PYME.recipes import processing
from PYME.IO.image import ImageStack
import numpy as np

def test_stats_by_frame():
    recipe = Recipe()
    test_length = 10
    x, y = np.meshgrid(range(test_length), range(test_length))
    mask = x > test_length/2  # mask out everything but 6, 7, 8, 9

    # check 2D
    recipe.namespace['input'] = ImageStack(data=x)
    recipe.namespace['mask'] = ImageStack(data=mask)
    stats_mod = processing.StatisticsByFrame(input_name='input', mask='mask', output_name='output')
    recipe.add_module(stats_mod)
    stats = recipe.execute()
    # check results
    assert len(stats['mean']) == 1
    assert stats['mean'] == 7.5

    # test 3D with 2D mask
    recipe.namespace.clear()
    x3, y3, z3 = np.meshgrid(range(test_length), range(test_length), range(test_length))
    recipe.namespace['input'] = ImageStack(data=z3)
    # reuse the same mask from before, which will now take the right 4 columns at each slice
    recipe.namespace['mask'] = ImageStack(data=mask)
    stats = recipe.execute()
    # check results
    np.testing.assert_array_almost_equal(stats['mean'], range(test_length))

    # test 3D with 3D mask
    mask = x3 > test_length / 2
    recipe.namespace['mask'] = ImageStack(data=mask)
    stats = recipe.execute()
    # check results
    np.testing.assert_array_almost_equal(stats['mean'], np.ma.masked_array(z3, mask=~(x3 > test_length / 2)).mean(axis=(0, 1)))

    # test no mask
    stats_mod.mask = ''
    stats = recipe.execute()
    np.testing.assert_array_almost_equal(stats['mean'], np.mean(z3, axis=(0, 1)))
