"""Joint planar EKF SLAM with AprilTag landmarks."""
from copy import deepcopy
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class MotionNoise:
    q_d_per_m: float = 0.0025
    q_theta_per_rad: float = 0.01
    q_theta_per_m: float = 0.01
    q_d_per_rad: float = 0.0004

    def __post_init__(self):
        rates = tuple(vars(self).values())
        if not np.all(np.isfinite(rates)) or min(rates) < 0:
            raise ValueError('Motion noise rates must be finite and nonnegative')


@dataclass(frozen=True)
class TagObservation:
    tag_id: int
    z: np.ndarray
    R: np.ndarray


@dataclass(frozen=True)
class UpdateResult:
    accepted: bool
    initialized: bool = False
    residual: np.ndarray = None


class EKFSLAM:
    def __init__(self, initial_pose=(0., 0., 0.), initial_covariance=None, noise_config=None):
        self.state = np.array(initial_pose, dtype=float, copy=True)
        if self.state.shape != (3,) or not np.all(np.isfinite(self.state)):
            raise ValueError('Initial pose must contain three finite values')
        self.state[2] = self.wrap_angle(self.state[2])

        self.P = np.zeros((3, 3))
        if initial_covariance is not None:
            P = np.asarray(initial_covariance, dtype=float)
            if P.shape != (3, 3) or not np.all(np.isfinite(P)) or not np.allclose(P, P.T):
                raise ValueError('Initial covariance must be finite, symmetric, and 3x3')
            P = (P + P.T) / 2
            if np.min(np.linalg.eigvalsh(P)) < -1e-12:
                raise ValueError('Initial covariance must be positive semidefinite')
            self.P = P

        self.landmark_map = {}  # tag_id -> index of landmark x in state
        self.landmark_indices = self.landmark_map  # Preserve original public attribute
        self.noise = MotionNoise() if noise_config is None else noise_config

    @staticmethod
    def wrap_angle(angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    # Predict the state and calculate motion Jacobians.
    def pred_state(self, state, v, w, dt):
        distance = v * dt
        turn = w * dt
        midpoint = state[2] + turn / 2
        c, s = np.cos(midpoint), np.sin(midpoint)

        pred = state.copy()
        pred[0] += distance * c
        pred[1] += distance * s
        pred[2] = self.wrap_angle(state[2] + turn)

        F = np.eye(len(state))
        F[0, 2] = -distance * s
        F[1, 2] = distance * c

        G = np.zeros((len(state), 2))
        G[:3] = ((c, -distance * s / 2),
                 (s, distance * c / 2),
                 (0, 1))
        return pred, F, G

    def calc_P(self, F, G, P, Q):
        predicted_P = F @ P @ F.T + G @ Q @ G.T
        return 0.5 * (predicted_P + predicted_P.T)

    def predict(self, v, omega, dt):
        if not np.all(np.isfinite((v, omega, dt))) or dt < 0:
            raise ValueError('Prediction requires finite inputs and nonnegative elapsed time')
        if dt == 0:
            return

        # Jacobians are evaluated at the old state.
        pred, F, G = self.pred_state(self.state, v, omega, dt)
        d, a = abs(v * dt), abs(omega * dt)
        Q = np.diag([
            self.noise.q_d_per_m * d + self.noise.q_d_per_rad * a,
            self.noise.q_theta_per_m * d + self.noise.q_theta_per_rad * a,
        ])
        self.P = self.calc_P(F, G, self.P, Q)
        self.state = pred

    # Initialize a new tag in map coordinates, preserving cross-covariance.
    def add_landmark(self, tag_id, z, R):
        x, y, theta = self.state[:3]
        c, s = np.cos(theta), np.sin(theta)
        rotation = np.array([[c, -s], [s, c]])
        landmark = np.array([x, y]) + rotation @ z
        n = len(self.state)

        J = np.zeros((2, n))
        J[:, :3] = ((1, 0, -s * z[0] - c * z[1]),
                    (0, 1,  c * z[0] - s * z[1]))

        cross_covariance = self.P @ J.T
        landmark_covariance = J @ self.P @ J.T + rotation @ R @ rotation.T
        expanded_P = np.block([
            [self.P, cross_covariance],
            [cross_covariance.T, landmark_covariance],
        ])
        self.landmark_map[tag_id] = n
        self.state = np.concatenate((self.state, landmark))
        self.P = 0.5 * (expanded_P + expanded_P.T)

    # Predict the tag's robot-frame position and measurement Jacobian.
    def calculate_z_pred_and_H(self, tag_id):
        i = self.landmark_map[tag_id]
        x, y, theta = self.state[:3]
        mx, my = self.state[i:i + 2]
        dx, dy = mx - x, my - y
        c, s = np.cos(theta), np.sin(theta)

        predicted_z = np.array([c * dx + s * dy, -s * dx + c * dy])
        H = np.zeros((2, len(self.state)))
        H[:, :3] = ((-c, -s, -s * dx + c * dy),
                    (s, -c, -c * dx - s * dy))
        H[:, i:i + 2] = ((c, s), (-s, c))
        return predicted_z, H

    def error(self, z, predicted_z):
        return z - predicted_z

    # Joseph-form covariance update.
    def kalman_update(self, z, predicted_z, H, R):
        residual = self.error(z, predicted_z)

        # H @ self.P @ H.T  gives the uncertainty of the predicted tag(s) and R adds in the camera uncertainty
        S = H @ self.P @ H.T + R
        K = np.linalg.solve(S, H @ self.P).T

        self.state += K @ residual
        self.state[2] = self.wrap_angle(self.state[2])

        I_KH = np.eye(len(self.state)) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        return UpdateResult(accepted=True, residual=residual)

    # Original public methods, retained for compatibility.
    def measurement_model(self, tag_id):
        return self.calculate_z_pred_and_H(tag_id)[0]

    def measurement_jacobian(self, tag_id):
        return self.calculate_z_pred_and_H(tag_id)[1]

    def process_observation(self, tag_id, z, R):
        try:
            z = np.asarray(z, dtype=float)
            R = np.asarray(R, dtype=float)
        except (ValueError, TypeError, OverflowError):
            return UpdateResult(accepted=False)

        if (not isinstance(tag_id, (int, np.integer)) or tag_id < 0
                or z.shape != (2,) or R.shape != (2, 2)
                or not np.all(np.isfinite(z)) or not np.all(np.isfinite(R))
                or not np.allclose(R, R.T)):
            return UpdateResult(accepted=False)

        R = (R + R.T) / 2
        try:
            np.linalg.cholesky(R)
        except np.linalg.LinAlgError:
            return UpdateResult(accepted=False)

        if tag_id not in self.landmark_map:
            self.add_landmark(tag_id, z, R)
            return UpdateResult(accepted=True, initialized=True)

        predicted_z, H = self.calculate_z_pred_and_H(tag_id)
        return self.kalman_update(z, predicted_z, H, R)

    def process_observations(self, observations):
        results = []
        seen = set()
        for obs in sorted(observations, key=lambda obs: obs.tag_id):
            if obs.tag_id in seen:
                continue
            result = self.process_observation(obs.tag_id, obs.z, obs.R)
            if result.accepted:
                seen.add(obs.tag_id)
            results.append(result)
        return results

    def get_robot_pose(self):
        return self.state[:3].copy()

    def get_robot_covariance(self):
        return self.P[:3, :3].copy()

    def get_landmarks(self):
        return {
            tag_id: (self.state[i:i + 2].copy(), self.P[i:i + 2, i:i + 2].copy())
            for tag_id, i in self.landmark_map.items()
        }

    def copy(self):
        return deepcopy(self)
