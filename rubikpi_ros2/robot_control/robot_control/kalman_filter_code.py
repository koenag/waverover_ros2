import numpy as np


class KalmanEKF:
    # Set up the state and the noise covariances.
    def __init__(self):
        self.state = np.zeros(3)  # [x, y, theta, mx1, my1, ...]
        self.P = np.zeros((3, 3))  # Initial pose is assumed known.
        self.landmark_map = {}  # tag_id (int) -> index of the landmark's x in state

        self.Q = np.diag([0.05**2, 0.05**2])  # noise on (v, w)
        self.R = np.diag([0.03**2, 0.03**2])  # noise on a tag's (x, y), robot frame

    # Keep an angle between -pi and pi.
    @staticmethod
    def wrap_angle(angle):
        return (angle + np.pi) % (2.0 * np.pi) - np.pi

    # Predict the pose and compute the state and control Jacobians.
    def pred_state(self, state, v, w, dt):
        theta = state[2]
        c, s = np.cos(theta), np.sin(theta)
        n = len(state)

        pred = state.copy()
        pred[0] += v * dt * c
        pred[1] += v * dt * s
        pred[2] = self.wrap_angle(theta + w * dt)

        F = np.eye(n)
        F[0, 2] = -v * dt * s
        F[1, 2] = v * dt * c

        G = np.zeros((n, 2))
        G[0, 0] = dt * c
        G[1, 0] = dt * s
        G[2, 1] = dt

        return pred, F, G

    # Propagate the covariance through the motion model.
    def calc_P(self, F, G, P, M):
        predicted_P = F @ P @ F.T + G @ M @ G.T
        return 0.5 * (predicted_P + predicted_P.T)

    # Initialize a landmark in world coordinates and expand its covariance.
    def add_landmark(self, tag_id, z):
        x, y, theta = self.state[:3]
        c, s = np.cos(theta), np.sin(theta)

        rotation = np.array([
            [c, -s],
            [s,  c],
        ])

        # add the distance between robot and tag to the currente estimated pose of the robot
        landmark = np.array([x, y]) + rotation @ z
        n = len(self.state)

        # Differentiate the new landmark position with respect to the state.
        J = np.zeros((2, n))
        J[:, :3] = np.array([
            [1.0, 0.0, -s * z[0] - c * z[1]],
            [0.0, 1.0,  c * z[0] - s * z[1]],
        ])

        cross_covariance = self.P @ J.T
        landmark_covariance = (
            J @ self.P @ J.T
            + rotation @ self.R @ rotation.T
        )

        expanded_P = np.zeros((n + 2, n + 2))
        expanded_P[:n, :n] = self.P
        expanded_P[:n, n:] = cross_covariance
        expanded_P[n:, :n] = cross_covariance.T
        expanded_P[n:, n:] = landmark_covariance

        self.landmark_map[tag_id] = n
        self.state = np.concatenate((self.state, landmark))
        self.P = 0.5 * (expanded_P + expanded_P.T)

    # Predict a landmark's position in the robot frame and compute its Jacobian.
    def z_pred(self, tag_id):
        i = self.landmark_map[tag_id]
        x, y, theta = self.state[:3]
        mx, my = self.state[i:i + 2]

        dx, dy = mx - x, my - y
        c, s = np.cos(theta), np.sin(theta)

        predicted_z = np.array([
            c * dx + s * dy,
            -s * dx + c * dy,
        ])

        H = np.zeros((2, len(self.state)))
        H[:, :3] = np.array([
            [-c, -s, -s * dx + c * dy],
            [ s, -c, -c * dx - s * dy],
        ])
        H[:, i:i + 2] = np.array([
            [ c, s],
            [-s, c],
        ])

        return predicted_z, H

    # Subtract the predicted observation from the measured observation.
    def error(self, z, predicted_z):
        return z - predicted_z

    # Correct the state and covariance using a known landmark.
    def kalman_update(self, z, predicted_z, H):
        residual = self.error(z, predicted_z)

        S = H @ self.P @ H.T + self.R
        K = np.linalg.solve(S, H @ self.P).T

        self.state += K @ residual
        self.state[2] = self.wrap_angle(self.state[2])

        I_KH = np.eye(len(self.state)) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
        self.P = 0.5 * (self.P + self.P.T)

    # Predict, then correct with any tags. tags: {tag_id: np.array([x, y])}
    # in the ROBOT frame. An empty dict means a prediction-only update.
    def update_state(self, v, w, dt, tags):
        self.state, F, G = self.pred_state(self.state, v, w, dt)
        self.P = self.calc_P(F, G, self.P, self.Q)

        for tag_id, z in tags.items():
            if tag_id not in self.landmark_map:
                self.add_landmark(tag_id, z)
            else:
                predicted_z, H = self.z_pred(tag_id)
                self.kalman_update(z, predicted_z, H)