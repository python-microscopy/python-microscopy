
from PYME.IO.tabular import DictSource
import numpy as np
from PYME.Analysis.points import multiview

def test_merge_clumps():
    # partially but not completely degenerate with points/test_clumps.py
    n = 10
    half_n = int(n/2)
    unmerged = DictSource({
        'x': np.arange(float(n)),
        'y': np.arange(float(n)),
        't': np.arange(n),
        'A': np.arange(float(n)),
        # error_y of 0 works, but do we want divide by zero warnings in test?
        'error_y': np.arange(1, float(n + 1)),
        # PYME cluster convention: 0 is unclustered, and we dodge it in 
        # multiview
        'cluster_id': np.concatenate([np.ones(half_n, int), 
                                      2 * np.ones(half_n, int)])
    })

    merged = multiview.merge_clumps(unmerged, 0, 'cluster_id')

    # test len
    assert len(merged['cluster_id']) == len(np.unique(unmerged['cluster_id']))

    # test sum
    np.testing.assert_allclose(merged['A'], 
                               [np.sum(unmerged['A'][:half_n]),
                                np.sum(unmerged['A'][half_n:])])

    # test mean
    np.testing.assert_allclose(merged['x'], 
                               [np.average(unmerged['x'][:half_n]),
                                np.average(unmerged['x'][half_n:])])

    # test weighted NOTE - this not simply weighted avg, we assume gasussian dist
    # could consider renaming if we start using aggregateWeightedMean in more
    # parts of the code.
    weights = 1 / (unmerged['error_y'] ** 2)
    wm = [
        np.sum(weights[:half_n] * unmerged['y'][:half_n]) / np.sum(weights[:half_n]),
        np.sum(weights[half_n:] * unmerged['y'][half_n:]) / np.sum(weights[half_n:])
    ]
    np.testing.assert_allclose(merged['y'], wm)
