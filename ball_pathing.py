import numpy as np
import random

MAX_SPEED = 500.0         # max speed in pixels/sec
ACCELERATION = 6.0        # how quickly the ball changes direction
TARGET_RADIUS = 40.0      # how close before picking a new target
RADIUS = 16


class Ball:
    def __init__(self, width, height):
        self.radius = RADIUS
        self.width = width
        self.height = height

        self.pos = self._random_position()
        self.target = self._random_position()
        self.velocity = np.zeros(2, dtype=float)

    def _random_position(self):
        return np.array([
            random.uniform(self.radius, self.width - self.radius),
            random.uniform(self.radius, self.height - self.radius)
        ], dtype=float)

    def update(self, dt, width, height):
        direction = self.target - self.pos
        distance = np.linalg.norm(direction)

        if distance < TARGET_RADIUS:
            self.target = self._random_position()
        else:
            desired_velocity = (direction / (distance + 1e-6)) * MAX_SPEED
            # Inertial smoothing toward desired velocity
            self.velocity += (desired_velocity - self.velocity) * ACCELERATION * dt
            self.pos += self.velocity * dt

        self.pos[0] = np.clip(self.pos[0], self.radius, width - self.radius)
        self.pos[1] = np.clip(self.pos[1], self.radius, height - self.radius)

    def draw(self, frame):
        y, x = np.ogrid[:frame.shape[0], :frame.shape[1]]
        dist = np.sqrt((x - self.pos[0])**2 + (y - self.pos[1])**2)
        mask = dist <= self.radius
        frame[mask] = [0, 255, 0, 255]
