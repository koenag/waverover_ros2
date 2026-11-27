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
        """Anchor the map at the initial pose, with zero uncertainty by default."""

        self.state = np.array(initial_pose, dtype=float, copy=True)
        self.state[2] = self.wrap_angle(self.state[2])

        self.P = np.diag([0.01, 0.01, 0.005]) # dont start cov at 0
        if initial_covariance is not None:
            cov = np.asarray(initial_covariance, dtype=float)
            self.P = (cov + cov.T) /2

        self.landmark_map = {}
        self.motion_noise = MotionNoise()

    @staticmethod
    def wrap_angle(angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    # Predict the state and calculate motion Jacobians.
    def pred_state(self, state, v, w, dt):
        distance = v*dt
        change_in_angle = w*dt
        facing_angle = change_in_angle/2 + state[2]
        c,s = np.cos(facing_angle), np.sin(facing_angle)

        pred=state.copy()
        pred[0] += distance*c
        pred[1] += distance*s
        pred[2] = self.wrap_angle(pred[2] + facing_angle)

        F = np.eye(len(self.state))
        F[0, 2] = (1,0, -distance*s)
        F[1, 2] = (0,1, distance*c)

        G = np.zeros((len(self.state), 2))
        G[:3] = ((c, -distance*s/2), (s, distance*c/2), (0,1))
        return pred, F, G
        


    def calc_P(self, F, G, P, Q):
        cov = F @ P @ F.T + G @ Q @ G.T
        return 0.5 * (cov + cov.T)

    def predict(self, v, omega, dt):
        if not np.all(np.isfinite((v, omega, dt))) or dt < 0:
            raise ValueError('Prediction requires finite inputs and nonnegative elapsed time')
        if dt == 0:
            return
        pred, F,G = self.pred_state(self.state, v, omega,dt)
        distance_travelled, angle_turned = abs(v*dt), abs(omega*dt)
        Q = np.diag([
            self.motion_noise.q_d_per_m*distance_travelled + self.motion_noise.q_d_per_rad*angle_turned,
            self.motion_noise.q_theta_per_m*distance_travelled, self.motion_noise.q_theta_per_rad*angle_turned
        ])

        self.P = self.calc_P(F,G,self.P,Q)
        self.state = pred

    # Initialize a new tag in map coordinates, preserving cross-covariance.
    def add_landmark(self, tag_id, z, R):
        x,y,theta = self.state[:3]
        c,s = np.cos(theta), np.sin(theta)
        rotation = np.array([[c,-s],[s, c]])
        landmark_init_pred = np.array([x,y]) + rotation @ z
        zx,zy = z

        J = np.zeros((2, len(self.state)))
        J[:, :3] = [1,0, -zx*s-zy*c], [0,1, zx*c - zy*s]

        #self.P @ J.T = how are errors in the robot and previously mapped landmarks correlated with errors in the new landmark's position?
                # it is used in the expanded P 
        
                #       [Cov(xr,    m2x)  Cov(xr,    m2y)]
                #       [Cov(yr,    m2x)  Cov(yr,    m2y)]
                #       [Cov(theta, m2x)  Cov(theta, m2y)]
                #       [Cov(m1x,   m2x)  Cov(m1x,   m2y)]
                #       [Cov(m1y,   m2x)  Cov(m1y,   m2y)]
        cross_covariance = self.P @ J.T

        # J @ self.P @ J.T =. how much is uncertainty of the landmark estimate position (J) affected by the uncertainty of the robots state (P)
                # rotation @ R @ rotation.T = rotate the camera prediction uncertainty from the robot frame into the world frame
                # landmark_covariance:
                #
                #       [Var(m2x)       Cov(m2x, m2y)]
                #       [Cov(m2y, m2x)  Var(m2y)     ]
        landmark_cov = J@self.P@J.T

                 # Expand P from (5 x 5) to (7 x 7):
                #
                #             Existing state (5)     New Tag 2 (2)
                #
                #           +---------------------+----------------+
                #           |                     |                |
                # Existing  |       P             | cross_cov      |
                # state (5) |     (5 x 5)         |   (5 x 2)      |
                #           |                     |                |
                #           +---------------------+----------------+
                #           |                     |                |
                # Tag 2 (2) |  cross_cov.T        | landmark_cov   |
                #           |     (2 x 5)         |    (2 x 2)     |
                #           +---------------------+----------------+
                #
        cov = np.block([[self.P, cross_covariance], [cross_covariance, landmark_cov]])

        self.P = 0.5 * (cov + cov.T)
        self.landmark_map[tag_id] = len(self.state)
        self.state = np.concatenate(self.state, landmark_init_pred)


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

    def get_covariance_matrix(self):
        return self.P

    def get_state_vector(self):
        return self.state

    def get_landmarks(self):
        return {
            tag_id: (self.state[i:i + 2].copy(), self.P[i:i + 2, i:i + 2].copy())
            for tag_id, i in self.landmark_map.items()
        }

    def copy(self):
        return deepcopy(self)
