"""Joint planar EKF for a robot and uniquely identified AprilTag centers."""

from copy import deepcopy
from dataclasses import dataclass
from math import cos, pi, sin

import numpy as np


def wrap_angle(angle):
    """Wrap a heading to [-pi, pi)."""
    return (angle + pi) % (2 * pi) - pi


@dataclass(frozen=True)
class MotionNoise:
    """Provisional variance growth rates for distance and heading increments."""

    # Translation and rotation can each add uncertainty to both distance and heading.
    # These coefficients are variance growth rates per unit of motion.
    q_d_per_m: float = 0.0025
    q_theta_per_rad: float = 0.01
    q_theta_per_m: float = 0.01
    q_d_per_rad: float = 0.0004

    def __post_init__(self):
        """Require finite, nonnegative growth rates."""
        rates = tuple(vars(self).values())
        if not np.all(np.isfinite(rates)) or min(rates) < 0:
            raise ValueError('Motion noise rates must be finite and nonnegative')


@dataclass(frozen=True)
class TagObservation:
    """A tag's forward/left coordinates and covariance in the robot frame."""

    tag_id: int
    z: np.ndarray
    R: np.ndarray


@dataclass(frozen=True)
class UpdateResult:
    """Describe initialization, correction, or rejection of one observation."""

    accepted: bool
    initialized: bool = False
    residual: np.ndarray = None


def motion_model(state, v, omega, dt):
    """Integrate the robot pose with midpoint heading; landmarks stay fixed."""
    distance = v * dt
    # Convert angular velocity to a heading increment for this interval.
    turn = omega * dt
    # Approximate the arc using the heading halfway through the turn.
    midpoint = state[2] + turn / 2
    # Only the first three entries describe the moving robot. The remaining
    # entries are tag centers in map coordinates and stay fixed in prediction.
    result = state.copy()
    result[0] += distance * cos(midpoint)
    result[1] += distance * sin(midpoint)
    result[2] = wrap_angle(state[2] + turn)
    return result


def motion_jacobians(state, v, omega, dt):
    """Linearize motion with respect to state and the two motion increments."""
    distance = v * dt
    midpoint = state[2] + omega * dt / 2
    c, s = cos(midpoint), sin(midpoint)
    # F maps old state errors to new state errors. Only the dependence of
    # robot x and y on the old heading changes this identity matrix.
    F = np.eye(len(state))
    F[:2, 2] = (-distance * s, distance * c)
    # G maps distance and turn errors into pose errors. The turn column
    # also affects x and y through the midpoint heading.
    G = np.zeros((len(state), 2))
    G[:3] = ((c, -distance * s / 2), (s, distance * c / 2), (0, 1))
    return F, G


class EKFSLAM:
    """Estimate the robot pose and landmarks in a shared map frame."""

    def __init__(self, initial_pose=(0., 0., 0.), initial_covariance=None, noise_config=None):
        """Anchor the map at the initial pose, with zero uncertainty by default."""
        # Start with [robot_x, robot_y, robot_yaw]. Append a pair of coordinates
        # for each new tag, with the covariance entries in the same order.
        self.state = np.array(initial_pose, dtype=float, copy=True)
        if self.state.shape != (3,) or not np.all(np.isfinite(self.state)):
            raise ValueError('Initial pose must contain three finite values')
        self.state[2] = wrap_angle(self.state[2])

        # Zero covariance anchors the startup pose as the map reference. Motion
        # and measurement noise determine uncertainty as the state evolves.
        self.P = np.zeros((3, 3))
        if initial_covariance is not None:
            # Accept a custom covariance only if it is a valid planar pose covariance.
            covariance = np.asarray(initial_covariance, dtype=float)
            if (
                covariance.shape != (3, 3)
                or not np.all(np.isfinite(covariance))
                or not np.allclose(covariance, covariance.T)
            ):
                raise ValueError('Initial covariance must be finite, symmetric, and 3x3')
            covariance = (covariance + covariance.T) / 2
            if np.min(np.linalg.eigvalsh(covariance)) < -1e-12:
                raise ValueError('Initial covariance must be positive semidefinite')
            self.P = covariance

        # Map each detector ID to the index of its first coordinate. IDs can
        # arrive in any order and need not be consecutive.
        self.landmark_indices = {}
        self.noise = MotionNoise() if noise_config is None else noise_config

    def predict(self, v, omega, dt):
        """Advance the joint estimate under one constant command."""
        # Check the time and command before touching the filter state.
        if not np.all(np.isfinite((v, omega, dt))) or dt < 0:
            raise ValueError('Prediction requires finite inputs and nonnegative elapsed time')
        # Events with equal timestamps share a state with no intervening motion.
        if dt == 0:
            return

        # Increment-dependent noise gives zero added uncertainty while stationary.
        distance, turn = abs(v * dt), abs(omega * dt)
        Q = np.diag((
            self.noise.q_d_per_m * distance + self.noise.q_d_per_rad * turn,
            self.noise.q_theta_per_m * distance + self.noise.q_theta_per_rad * turn,
        ))
        # Linearize about the old pose. F carries existing uncertainty and
        # correlations forward. G adds uncertainty from this motion interval.
        F, G = motion_jacobians(self.state, v, omega, dt)
        covariance = F @ self.P @ F.T + G @ Q @ G.T
        # Advance the mean, then symmetrize covariance to remove roundoff drift.
        self.state = motion_model(self.state, v, omega, dt)
        self.P = (covariance + covariance.T) / 2

    def get_robot_pose(self):
        """Return the robot's x, y, and yaw."""
        return self.state[:3].copy()

    def get_robot_covariance(self):
        """Return the robot's planar marginal covariance."""
        return self.P[:3, :3].copy()

    def get_landmarks(self):
        """Return landmark means and marginal covariances by tag ID."""
        return {
            # Return the position and its own 2x2 covariance for each tag.
            # The full matrix and its cross-correlations stay inside the filter.
            tag_id: (self.state[i:i + 2].copy(), self.P[i:i + 2, i:i + 2].copy())
            for tag_id, i in self.landmark_indices.items()
        }

    def copy(self):
        """Return an independent estimate for temporary forward prediction."""
        # Give the ROS node independent arrays and tag indices so live-pose
        # extrapolation leaves the saved filter untouched.
        return deepcopy(self)

    def measurement_model(self, tag_id):
        """Predict the tag center in robot-relative forward/left coordinates."""
        i = self.landmark_indices[tag_id]
        # Find the displacement in map axes, then rotate by minus the robot yaw
        # to express it in forward/left axes.
        dx, dy = self.state[i:i + 2] - self.state[:2]
        c, s = cos(self.state[2]), sin(self.state[2])
        return np.array((c * dx + s * dy, -s * dx + c * dy))

    def measurement_jacobian(self, tag_id):
        """Linearize the predicted measurement at the current estimate."""
        i = self.landmark_indices[tag_id]
        c, s = cos(self.state[2]), sin(self.state[2])
        forward, left = self.measurement_model(tag_id)
        # Only the robot and the observed tag enter this model. The heading
        # derivatives are [left, -forward]. Other columns stay zero.
        H = np.zeros((2, len(self.state)))
        H[:, :3] = ((-c, -s, left), (s, -c, -forward))
        H[:, i:i + 2] = ((c, s), (-s, c))
        return H

    def _initialize_landmark(self, tag_id, z, R):
        # Rotate the observation into the map and propagate pose uncertainty.
        c, s = cos(self.state[2]), sin(self.state[2])
        rotation = np.array(((c, -s), (s, c)))
        offset = rotation @ z
        # J maps errors in the existing state into the new tag position.
        # Only the robot pose enters the coordinate transform directly.
        J = np.zeros((2, len(self.state)))
        J[:, :3] = ((1, 0, -offset[1]), (0, 1, offset[0]))

        # The new tag inherits correlations with every existing state entry.
        cross = self.P @ J.T
        # Add the new tag uncertainty from both robot pose and camera noise.
        # Rotate the camera measurement covariance from robot to map axes.
        marginal = J @ cross + rotation @ R @ rotation.T
        # Append the tag block and its correlations without replacing the
        # covariance of the existing state.
        covariance = np.block([[self.P, cross], [cross.T, marginal]])
        i = len(self.state)
        self.state = np.concatenate((self.state, self.state[:2] + offset))
        self.P = (covariance + covariance.T) / 2
        self.landmark_indices[tag_id] = i

    def _update_landmark(self, tag_id, z, R):
        # The innovation is Cartesian, so only the updated heading is wrapped.
        residual = z - self.measurement_model(tag_id)
        H = self.measurement_jacobian(tag_id)
        # Project uncertainty into measurement space, add sensor noise, then
        # solve for the Kalman gain without forming a matrix inverse.
        PHt = self.P @ H.T
        innovation_covariance = H @ PHt + R
        gain = np.linalg.solve(innovation_covariance, PHt.T).T
        # The gain covers the entire state. Correlated tags can move even if
        # they were absent from this camera frame.
        state = self.state + gain @ residual
        state[2] = wrap_angle(state[2])

        # Joseph form preserves covariance symmetry and numerical stability.
        A = np.eye(len(state)) - gain @ H
        covariance = A @ self.P @ A.T + gain @ R @ gain.T
        self.state = state
        self.P = (covariance + covariance.T) / 2
        return UpdateResult(accepted=True, residual=residual)

    def process_observation(self, tag_id, z, R):
        """Validate an observation, then initialize or correct its landmark."""
        try:
            # Convert to arrays so shape and numeric checks run before any update.
            z, R = np.asarray(z, dtype=float), np.asarray(R, dtype=float)
        except (ValueError, TypeError, OverflowError):
            return UpdateResult(accepted=False)
        if (
            not isinstance(tag_id, (int, np.integer))
            or tag_id < 0
            or z.shape != (2,)
            or R.shape != (2, 2)
            or not np.all(np.isfinite(z))
            or not np.all(np.isfinite(R))
            or not np.allclose(R, R.T)
        ):
            return UpdateResult(accepted=False)

        # Cholesky verifies that R is positive definite. With a valid state
        # covariance, this keeps the innovation system nonsingular.
        R = (R + R.T) / 2
        try:
            np.linalg.cholesky(R)
        except np.linalg.LinAlgError:
            return UpdateResult(accepted=False)

        # Reobservations correct the map. The first sighting initializes a tag
        # and its covariance without using the same observation twice.
        if tag_id in self.landmark_indices:
            return self._update_landmark(tag_id, z, R)
        self._initialize_landmark(tag_id, z, R)
        return UpdateResult(accepted=True, initialized=True)

    def process_observations(self, observations):
        """Apply at most one valid observation per tag from a camera frame."""
        results = []
        seen = set()
        # Use a consistent tag order for sequential updates. Mark an ID seen
        # only after accepting its observation.
        for obs in sorted(observations, key=lambda observation: observation.tag_id):
            if obs.tag_id in seen:
                continue
            result = self.process_observation(obs.tag_id, obs.z, obs.R)
            if result.accepted:
                seen.add(obs.tag_id)
            results.append(result)
        return results
