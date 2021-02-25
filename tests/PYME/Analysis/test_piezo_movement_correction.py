
import numpy as np
from PYME.IO.MetaDataHandler import NestedClassMDHandler

# basePiezo position - offset = OffsetPiezo position
# Note that the ProtocolFocus events know about the offset (ie are logged by
# the OffsetPiezo), while the PiezoOnTarget events are logged by the BasePiezo.

TEST_EVENTS = np.array([('PiezoOnTarget', 0, '50.000'),
       ('ProtocolFocus', 0, '0, 49.500'),
       ('PiezoOnTarget', 1 * 0.00125, '50.001'),
       ('ProtocolTask', 1 * 0.00125, '1, DisableLock, '),
       ('PiezoOffsetUpdate', 2 * 0.00125, '-0.5'),
       ('ProtocolFocus', 801 * 0.00125, '801, 51.000'),
       ('PiezoOnTarget', 850 * 0.00125, '50.600'),
       ('ProtocolFocus', 1601 * 0.00125, '1601, 52.000'),
       ('PiezoOnTarget', 1650 * 0.00125, '51.450'),
       ('ProtocolFocus', 2401 * 0.00125, '2401, 53'),
       ('PiezoOnTarget', 2450 * 0.00125, '52.705'),
       ('ProtocolTask', 2501 * 0.00125, 'EnableLock, '),
       ('ProtocolTask', 2501 * 0.00125, 'LaunchAnalysis, ')],
                       # fixme - but the S32 and S256 back to unicode once we fix event typing elsewhere
      dtype=[('EventName', 'S32'), ('Time', '<f8'), ('EventDescr', 'S256')])

CHANGES = np.array([(0, 50.0 + 0.5),
                    (1, 50.001 + 0.5),
                    (801, 51.000),
                    (850, 50.600 + 0.5),
                    (1601, 52.000),
                    (1650, 51.450 + 0.5),
                    (2401, 53),
                    (2450,  52.705 + 0.5)],
                    dtype= [('frame', '<i4'), ('z', '<f8')])

TEST_DATA_SOURCE = np.arange(2500).astype([('t', '<i4')])
GROUND_TRUTH_Z = np.empty(len(TEST_DATA_SOURCE), dtype=float)
for ind in range(len(CHANGES)):
    GROUND_TRUTH_Z[CHANGES[ind]['frame']:] = CHANGES[ind]['z'] 

TEST_MDH = NestedClassMDHandler()
TEST_MDH['Camera.CycleTime'] = 0.00125
TEST_MDH['StartTime'] = 0

def test_flag_piezo_movement():
    from PYME.Analysis.piezo_movement_correction import flag_piezo_movement

    moving = flag_piezo_movement(TEST_DATA_SOURCE['t'], TEST_EVENTS, TEST_MDH)
    assert np.all(moving[np.where(np.logical_and(TEST_DATA_SOURCE['t'] >= 2401,
                                                 TEST_DATA_SOURCE['t'] < 2450))])
    assert not np.all(np.all(moving[np.where(TEST_DATA_SOURCE['t'] >= 2450)]))

def test_focus_correction():
    from PYME.Analysis.piezo_movement_correction import correct_target_positions
    corrected_focus = correct_target_positions(TEST_DATA_SOURCE['t'], TEST_EVENTS, TEST_MDH)
    np.testing.assert_array_almost_equal(GROUND_TRUTH_Z, corrected_focus)
