"""Regression tests for the ROS-independent EKF."""

import numpy as np
from numpy.testing import assert_allclose
from robot_control.ekf_slam import EKFSLAM, TagObservation, motion_jacobians, motion_model


def test_motion_jacobians():
    """Check the midpoint model's state and increment derivatives."""
    state = np.array([0.2, -0.1, 0.3, 2.0, 1.0])
    v, omega, dt = 0.21, -0.6, 0.18
    epsilon = 1e-6
    F, G = motion_jacobians(state, v, omega, dt)
    for index, delta in enumerate(np.eye(len(state)) * epsilon):
        plus = motion_model(state + delta, v, omega, dt)
        minus = motion_model(state - delta, v, omega, dt)
        assert_allclose((plus - minus) / (2 * epsilon), F[:, index], atol=1e-9)
    for index, (dv, dw) in enumerate(np.eye(2) * epsilon / dt):
        plus = motion_model(state, v + dv, omega + dw, dt)
        minus = motion_model(state, v - dv, omega - dw, dt)
        assert_allclose((plus - minus) / (2 * epsilon), G[:, index], atol=1e-9)


def test_landmark_augmentation_and_correction():
    """Verify cross-covariance and correction of an unobserved landmark."""
    covariance = np.diag([0.02, 0.03, 0.01])
    ekf = EKFSLAM(initial_pose=(1, 0, np.pi / 2), initial_covariance=covariance)
    R = np.eye(2) * 0.01
    assert ekf.process_observation(7, [2, 0], R).initialized
    assert_allclose(ekf.get_landmarks()[7][0], [1, 2])
    expected_cross = covariance @ np.array([[1, 0, -2], [0, 1, 0]]).T
    assert_allclose(ekf.P[:3, 3:5], expected_cross, atol=1e-12)
    ekf.predict(0.3, 0.25, 1.0)
    assert ekf.process_observation(12, [3, 1], R).initialized

    jacobian = ekf.measurement_jacobian(7)
    original = ekf.state.copy()
    epsilon = 1e-6
    for index, delta in enumerate(np.eye(len(original)) * epsilon):
        ekf.state = original + delta
        plus = ekf.measurement_model(7)
        ekf.state = original - delta
        minus = ekf.measurement_model(7)
        assert_allclose((plus - minus) / (2 * epsilon), jacobian[:, index], atol=1e-9)
    ekf.state = original

    before = ekf.get_landmarks()[12][0]
    z = ekf.measurement_model(7) + [0.1, -0.1]
    assert ekf.process_observation(7, z, R).accepted
    assert not np.allclose(before, ekf.get_landmarks()[12][0])
    assert_allclose(ekf.P, ekf.P.T)
    assert np.min(np.linalg.eigvalsh(ekf.P)) >= -1e-12


def test_stationary_frame_deduplication_and_copy():
    """Keep the stationary anchor fixed and isolate copy-only predictions."""
    ekf = EKFSLAM()
    observation = TagObservation(6, np.array([2., 0.]), np.eye(2) * 0.04)
    assert len(ekf.process_observations([observation, observation])) == 1
    assert not ekf.process_observation(5, [2, 0], np.zeros((2, 2))).accepted
    assert not ekf.process_observation(5, [2, np.nan], np.eye(2)).accepted
    before = ekf.get_landmarks()[6][1]
    assert ekf.process_observations([observation])[0].accepted
    assert_allclose(ekf.get_landmarks()[6][1], before / 2)
    assert_allclose(ekf.get_robot_covariance(), 0)

    temporary = ekf.copy()
    temporary.predict(-0.2, 0.0, 2.0)
    temporary.process_observation(10, [1, 0], np.eye(2))
    assert_allclose(temporary.get_robot_pose(), [-0.4, 0, 0])
    assert_allclose(ekf.get_robot_pose(), 0)
    assert set(ekf.get_landmarks()) == {6}
