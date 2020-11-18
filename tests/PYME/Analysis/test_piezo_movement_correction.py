
import numpy as np
from PYME.IO.MetaDataHandler import NestedClassMDHandler

# basePiezo position - offset = OffsetPiezo position
# Note that the ProtocolFocus events know about the offset (ie are logged by
# the OffsetPiezo), while the PiezoOnTarget events are logged by the BasePiezo.

TEST_EVENTS = np.array([('PiezoOnTarget', 0, '48.262'),
       ('ProtocolFocus', 0, '0, 49.988'),
       ('PiezoOnTarget', 1 * 0.00125, '48.307'),
       ('ProtocolTask', 1 * 0.00125, '1, DisableLock, '),
       ('PiezoOffsetUpdate', 2 * 0.00125, '-1.6720'),
       ('ProtocolFocus', 801 * 0.00125, '801, 51.188'),
       ('PiezoOnTarget', 850 * 0.00125, '49.489'),
       ('ProtocolFocus', 1601 * 0.00125, '1601, 52.388'),
       ('PiezoOnTarget', 1650 * 0.00125, '50.705'),
       ('ProtocolFocus', 2401 * 0.00125, '2401, 53.588'),
       ('ProtocolTask', 2501 * 0.00125, 'EnableLock, '),
       ('ProtocolTask', 2501 * 0.00125, 'LaunchAnalysis, ')],
                       # fixme - but the S32 and S256 back to unicode once we fix event typing elsewhere
      dtype=[('EventName', 'S32'), ('Time', '<f8'), ('EventDescr', 'S256')])

CHANGES = np.array([(0, 0, 48.262, 48.262),
       (1 * 0.00125, 1, 48.307, 48.307),
       (801 * 0.00125, 801, 51.188, 51.188),
       (850 * 0.00125, 850, 49.489 + 1.6720, 49.489),
       (1601 * 0.00125, 1601, 52.388, 52.388),
       (1650 * 0.00125, 1650, 50.705 + 1.6720, 50.705),
       (2401 * 0.00125, 2401, 53.588, 53.588)],
       dtype= [('t', '<f8'), ('frame', '<i4'), ('z', '<f8'), 
               ('z_ignoring_offset', '<f8')])

TEST_DATA_SOURCE = np.arange(2500).astype([('t', '<i4')])
GROUND_TRUTH_Z = np.empty(len(TEST_DATA_SOURCE), dtype=float)
GROUND_TRUTH_Z_IGNORING_OFFSET = np.empty_like(GROUND_TRUTH_Z)
I = np.argsort(CHANGES['t'])
CHANGES = CHANGES[I]
for frame, z, z_no_off in zip(CHANGES['frame'], CHANGES['z'], CHANGES['z_ignoring_offset']):
    GROUND_TRUTH_Z[frame:] = z
    GROUND_TRUTH_Z_IGNORING_OFFSET[frame:] = z_no_off


TEST_MDH = NestedClassMDHandler()
TEST_MDH['Camera.CycleTime'] = 0.00125
TEST_MDH['StartTime'] = 0

def test_flag_piezo_movement():
    from PYME.Analysis.piezo_movement_correction import flag_piezo_movement

    moving = flag_piezo_movement(TEST_DATA_SOURCE['t'], TEST_EVENTS, TEST_MDH)
    assert np.all(moving[np.where(TEST_DATA_SOURCE['t'] >= 2401)])
    assert not np.all(np.all(moving[np.where(TEST_DATA_SOURCE['t'] >= 2400)]))

def test_focus_correction():
    from PYME.Analysis.piezo_movement_correction import correct_target_positions
    corrected_focus = correct_target_positions(TEST_DATA_SOURCE['t'], TEST_EVENTS, TEST_MDH)
    np.testing.assert_array_almost_equal(GROUND_TRUTH_Z, corrected_focus)
