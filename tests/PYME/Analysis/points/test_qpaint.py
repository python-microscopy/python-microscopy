"""
Tests for PYME.Analysis.points.qpaint

The off-time (dark time) distribution of a diffraction limited spot containing N
docking sites is exponential with a mean of ``tau_dark = 1/(N*k_on*c)``, so the
qPAINT index ``1/tau_dark`` is proportional to N. The tests below check both the
mechanics of building the empirical CDF and that the fitted qPAINT index scales 
with the number of docking sites.
"""
import numpy as np
import pytest

from PYME.Analysis.points import qpaint


def _binding_frames(tau_dark, tau_bright, n_frames, rng, frame_time=1.0):
    """
    Frame numbers in which a single docking site has an imager bound.

    Dark times and bright times are both exponentially distributed. 
    A frame counts as a detection if an imager is bound at any point during it.

    Parameters
    ----------
    tau_dark, tau_bright : float
        Mean dark and bright times, in the same units as `frame_time`.
    n_frames : int
        Length of the simulated series.
    rng : numpy.random.Generator
    frame_time : float
        Camera frame time.

    Returns
    -------
    ndarray of int
        Sorted, unique frame numbers which contain a "detection"
    """
    duration = n_frames * frame_time
    frames = []

    t = rng.exponential(tau_dark)  # start dark, so we don't bias the first gap
    while t < duration:
        t_off = t + rng.exponential(tau_bright)
        first = int(np.floor(t / frame_time))
        last = min(int(np.floor(t_off / frame_time)), n_frames - 1)
        frames.extend(range(first, last + 1))

        t = t_off + rng.exponential(tau_dark)

    return np.unique(np.array(frames, dtype='i'))


def _multi_site_frames(n_sites, tau_dark_site, tau_bright, n_frames, rng, frame_time=1.0):
    """
    Frame numbers for `n_sites` independent docking sites in one spot.
    Returns
    -------
    ndarray of int
        Sorted, unique frame numbers which contain or OR more "detections"
    """
    frames = [_binding_frames(tau_dark_site, tau_bright, n_frames, rng, frame_time)
              for _ in range(n_sites)]

    return np.unique(np.concatenate(frames))


# ---------------------------------------------------------------- off-time CDF

def test_offtime_cdf_simple():
    # gaps (in blank frames) are 0, 0, 2, 0, 3 -> two dark times, [2, 3]
    x, y = qpaint.get_offtime_cdf(np.array([0, 1, 2, 5, 6, 10]))

    assert np.allclose(x, [2, 3])
    assert np.allclose(y, [0., 0.5])


def test_offtime_cdf_merges_consecutive_frames():
    """Detections in consecutive frames are one binding event, not several."""
    x_long, y_long = qpaint.get_offtime_cdf(np.array([0, 1, 2, 3, 10, 11]))
    x_short, y_short = qpaint.get_offtime_cdf(np.array([3, 10]))

    assert np.allclose(x_long, x_short)
    assert np.allclose(y_long, y_short)
    assert np.allclose(x_long, [6])


def test_offtime_cdf_ignores_input_order():
    frames = np.array([0, 1, 2, 5, 6, 10, 30])
    x, y = qpaint.get_offtime_cdf(frames)

    shuffled = frames.copy()
    np.random.RandomState(42).shuffle(shuffled)
    x_s, y_s = qpaint.get_offtime_cdf(shuffled)

    assert np.allclose(x, x_s)
    assert np.allclose(y, y_s)


def test_offtime_cdf_envelopes():
    # gaps are 2, 2, 5 - the repeated value of 2 is what makes the two envelopes
    # differ
    frames = np.array([0, 3, 6, 12])

    x_u, y_u = qpaint.get_offtime_cdf(frames, envelope='upper')
    x_l, y_l = qpaint.get_offtime_cdf(frames, envelope='lower')

    assert np.allclose(x_u, [2, 5])
    assert np.allclose(x_l, [2, 5])

    assert np.allclose(y_u, [1. / 3, 2. / 3])
    assert np.allclose(y_l, [0., 2. / 3])

    assert np.all(y_u >= y_l)


def test_offtime_cdf_is_a_cdf():
    rng = np.random.default_rng(1234)
    frames = _binding_frames(50., 3., 5000, rng)

    for envelope in ['upper', 'lower']:
        x, y = qpaint.get_offtime_cdf(frames, envelope=envelope)

        assert len(x) == len(y)
        assert np.all(np.diff(x) > 0)  # strictly increasing, no repeated x values
        assert np.all(x >= 1)  # a dark time of zero frames is not a gap
        assert np.all(np.diff(y) > 0)  # monotonic
        assert y.min() >= 0.
        assert y.max() < 1.

# --------------------------------------------------------------- model functions

def test_qpaint1mod():
    tau = 137.
    p = np.sqrt([tau])
    t = np.array([0., 1., 10., 100., 1e6])

    assert np.allclose(qpaint.qpaint1mod(p, t), 1 - np.exp(-t / tau))
    assert qpaint.qpaint1mod(p, np.array([0.]))[0] == 0.
    assert np.allclose(qpaint.qpaint1mod(p, np.array([1e6]))[0], 1.)

    # parameters are squared internally, so the sign of p should not matter
    assert np.allclose(qpaint.qpaint1mod(-p, t), qpaint.qpaint1mod(p, t))


def test_qpaint2mod():
    a, t1, t2 = 0.3, 5., 400.
    p = np.sqrt([a, t1, t2])
    t = np.array([0., 1., 10., 100., 1e6])

    expected = 1 - a * np.exp(-t / t1) - (1 - a) * np.exp(-t / t2)
    assert np.allclose(qpaint.qpaint2mod(p, t), expected)

    assert qpaint.qpaint2mod(p, np.array([0.]))[0] == 0.
    assert np.allclose(qpaint.qpaint2mod(p, np.array([1e9]))[0], 1.)

    assert np.allclose(qpaint.qpaint2mod(-p, t), qpaint.qpaint2mod(p, t))


def test_qpaint2mod_degenerates_to_qpaint1mod():
    """With both time constants equal, the 2-component model is 1-component."""
    tau = 250.
    t = np.logspace(0, 4, 50)

    two = qpaint.qpaint2mod(np.sqrt([0.4, tau, tau]), t)
    one = qpaint.qpaint1mod(np.sqrt([tau]), t)

    assert np.allclose(two, one)


# ------------------------------------------------------------------------ fits

def test_start_params_track_the_data():
    """
    Start parameters are derived from the median of the CDF rather than fixed, so
    that short dark times don't start the fit orders of magnitude away.
    """
    for tau in [5., 50., 5000.]:
        t = np.logspace(0, np.log10(20 * tau), 200)
        c = qpaint.qpaint1mod(np.sqrt([tau]), t)

        t1, = qpaint._start_params(t, c, 'q1') ** 2
        assert t1 == pytest.approx(tau, rel=0.1)

        a, t1, t2 = qpaint._start_params(t, c, 'q2') ** 2
        assert t2 == pytest.approx(tau, rel=0.1)
        assert 0 < t1 < t2 or t1 == 1.  # fast component starts below the slow one
        assert 0 < a < 1


def test_start_params_stay_positive():
    """Degenerate CDFs must not produce a zero or negative start value."""
    t = np.array([1., 2.])
    c = np.array([0.9, 0.95])  # median below the smallest dark time

    assert np.all(qpaint._start_params(t, c, 'q1') ** 2 > 0)
    assert np.all(qpaint._start_params(t, c, 'q2') ** 2 > 0)


@pytest.mark.parametrize('tau_dark', [20., 100., 500.])
def test_fit_single_component_recovers_tau(tau_dark):
    rng = np.random.default_rng(7)
    frames = _binding_frames(tau_dark, 3., int(400 * tau_dark), rng)

    tau_fit, = qpaint.fit(frames, model='q1')

    assert tau_fit == pytest.approx(tau_dark, rel=0.15)


@pytest.mark.parametrize('seed', range(8))
def test_fit_converges_for_short_dark_times(seed):
    """
    Regression test - with a fixed start value of 500 frames the single-component
    fit would occasionally fail to converge and collapse to tau ~ 0 for dark times
    of a few tens of frames.
    """
    rng = np.random.default_rng(seed)
    frames = _binding_frames(20., 3., 8000, rng)

    tau_fit, = qpaint.fit(frames, model='q1')

    assert tau_fit == pytest.approx(20., rel=0.3)


def test_fit_two_component_recovers_both_taus():
    """A mixture of a fast and a slow population, e.g. two unresolved objects."""
    rng = np.random.default_rng(11)

    fast = _binding_frames(20., 2., 40000, rng)
    slow = _binding_frames(600., 2., 400000, rng)

    # concatenating the (well separated) dark times of the two populations rather
    # than merging two traces keeps the mixing fraction exactly known
    dark_times = np.hstack([np.diff(fast) - 1, np.diff(slow) - 1])
    dark_times = dark_times[dark_times > 0]
    frames = np.cumsum(dark_times + 1)

    a, t1, t2 = qpaint.fit(frames, model='q2')

    n_fast = (np.diff(fast) - 1 > 0).sum()
    n_slow = (np.diff(slow) - 1 > 0).sum()
    a_expected = n_fast / float(n_fast + n_slow)

    assert t1 == pytest.approx(20., rel=0.25)
    assert t2 == pytest.approx(600., rel=0.25)
    assert a == pytest.approx(a_expected, abs=0.1)


def test_qpaint_index_scales_with_number_of_docking_sites():
    """
    for N docking sites in one diffraction limited spot, the qPAINT index 1/tau_dark is proportional to N.

    The test simulates 1, 2 and 4 docking sites and checks that the fitted qPAINT index
    scales linearly.
    """
    rng = np.random.default_rng(1)

    tau_dark_site = 400.  # frames, for a single site
    tau_bright = 2.
    n_frames = 400000

    n_sites = [1, 2, 4]
    q_index = []
    for n in n_sites:
        frames = _multi_site_frames(n, tau_dark_site, tau_bright, n_frames, rng)
        tau_fit, = qpaint.fit(frames, model='q1')
        q_index.append(1. / tau_fit)

    # normalize to the N=1 calibration
    n_estimated = np.array(q_index) / q_index[0]

    assert np.allclose(n_estimated, n_sites, rtol=0.2)
