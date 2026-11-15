import numpy as np
import pytest

from vitesse import (ClusteredPositionSampler, GaussianNoise, MeasurementSimulator,
                     SpeedProfile)

PROFILE = SpeedProfile([25.0, 3.0, -8.0], length=2000.0)


def make(noise=None, **kw):
    return MeasurementSimulator(PROFILE, noise or GaussianNoise(2.0),
                                rng=np.random.default_rng(0), **kw)


def test_sizes_default_100_points_70_vehicles():
    m = make().simulate()
    assert m.n_points == 100
    assert m.n_vehicles == 70


def test_positions_within_segment():
    m = make().simulate()
    assert m.x_m.min() >= 0.0 and m.x_m.max() <= PROFILE.length


def test_every_vehicle_sends_at_least_once():
    m = make(n_vehicles=30, n_points=31).simulate()
    assert set(m.vehicle_ids) == set(range(30))


def test_noise_free_returns_true_profile():
    m = make(GaussianNoise(0.0)).simulate()
    np.testing.assert_allclose(m.v, PROFILE(m.x_m))


def test_measure_keeps_positions_but_redraws_noise():
    sim = make()
    x, ids = sim.draw_geometry()
    a, b = sim.measure(x, ids), sim.measure(x, ids)
    np.testing.assert_array_equal(a.x_m, b.x_m)
    assert not np.allclose(a.v, b.v)


def test_clustered_sampler_leaves_a_gap():
    m = make(position_sampler=ClusteredPositionSampler(0.0, 0.5)).simulate()
    assert m.x_m.max() <= 0.5 * PROFILE.length


def test_invalid_counts():
    with pytest.raises(ValueError):
        make(n_vehicles=10, n_points=5)
    with pytest.raises(ValueError):
        ClusteredPositionSampler(0.7, 0.3)
