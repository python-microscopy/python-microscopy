
import numpy as np
from PYME.IO.MetaDataHandler import NestedClassMDHandler

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

TEST_DATA_SOURCE = np.arange(2500).astype([('t', '<i4')])

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
    # fixme - write an actual test
