"""Eight-state constant-velocity Kalman filter for bounding boxes."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..config import KalmanConfig
from ..types import BBox

FloatArray = NDArray[np.float64]


class BoundingBoxKalmanFilter:
    """Estimate ``[cx, cy, w, h]`` and their four velocities."""

    state_size = 8
    measurement_size = 4

    def __init__(self, bbox: BBox, config: KalmanConfig) -> None:
        self.config = config
        self.x: FloatArray = np.zeros((self.state_size, 1), dtype=np.float64)
        self.x[:4, 0] = np.asarray(bbox.cxcywh, dtype=np.float64)

        position_variance = float(config.initial_position_variance)
        velocity_variance = float(config.initial_velocity_variance)
        self.P: FloatArray = np.diag(
            [position_variance] * 4 + [velocity_variance] * 4
        ).astype(np.float64)
        self.H: FloatArray = np.concatenate(
            (np.eye(4, dtype=np.float64), np.zeros((4, 4), dtype=np.float64)), axis=1
        )
        self.R: FloatArray = np.diag(
            [
                config.measurement_position_variance,
                config.measurement_position_variance,
                config.measurement_size_variance,
                config.measurement_size_variance,
            ]
        ).astype(np.float64)
        self._identity: FloatArray = np.eye(self.state_size, dtype=np.float64)

    @staticmethod
    def _transition(dt: float) -> FloatArray:
        transition = np.eye(8, dtype=np.float64)
        transition[:4, 4:] = np.eye(4, dtype=np.float64) * dt
        return transition

    def _process_noise(self, dt: float) -> FloatArray:
        noise = np.zeros((8, 8), dtype=np.float64)
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2
        for position_index in range(4):
            velocity_index = position_index + 4
            variance = (
                self.config.process_position_noise
                if position_index < 2
                else self.config.process_size_noise
            )
            noise[position_index, position_index] = variance * dt4 / 4.0
            noise[position_index, velocity_index] = variance * dt3 / 2.0
            noise[velocity_index, position_index] = variance * dt3 / 2.0
            noise[velocity_index, velocity_index] = variance * dt2
        return noise

    def predict(self, dt: float) -> BBox:
        if dt <= 0.0:
            raise ValueError("Kalman time step must be positive")
        transition = self._transition(dt)
        self.x = transition @ self.x
        self.P = transition @ self.P @ transition.T + self._process_noise(dt)
        self._stabilize_size()
        return self.bbox

    def update(self, bbox: BBox) -> BBox:
        measurement = np.asarray(bbox.cxcywh, dtype=np.float64).reshape(4, 1)
        innovation = measurement - self.H @ self.x
        innovation_covariance = self.H @ self.P @ self.H.T + self.R
        cross_covariance = self.P @ self.H.T
        gain = np.linalg.solve(innovation_covariance.T, cross_covariance.T).T

        self.x = self.x + gain @ innovation
        # Joseph form is more numerically stable and preserves symmetry.
        correction = self._identity - gain @ self.H
        self.P = correction @ self.P @ correction.T + gain @ self.R @ gain.T
        self.P = (self.P + self.P.T) / 2.0
        self._stabilize_size()
        return self.bbox

    def project(self) -> tuple[FloatArray, FloatArray]:
        mean = self.H @ self.x
        covariance = self.H @ self.P @ self.H.T + self.R
        return mean, covariance

    def mahalanobis_squared(self, bbox: BBox) -> float:
        measurement = np.asarray(bbox.cxcywh, dtype=np.float64).reshape(4, 1)
        mean, covariance = self.project()
        delta = measurement - mean
        try:
            solved = np.linalg.solve(covariance, delta)
        except np.linalg.LinAlgError:
            solved = np.linalg.pinv(covariance) @ delta
        return float((delta.T @ solved).item())

    @property
    def bbox(self) -> BBox:
        cx, cy, width, height = self.x[:4, 0]
        return BBox.from_cxcywh(float(cx), float(cy), float(width), float(height))

    def _stabilize_size(self) -> None:
        self.x[2, 0] = max(self.x[2, 0], 1.0)
        self.x[3, 0] = max(self.x[3, 0], 1.0)
