# ball_pathing.py
import numpy as np
import random

VELOCITY = 256
HOLD_DURATION = 5.0  # seconds


class Ball:
    def __init__(self, width, height):
        self.radius = 30 * 0.85
        self.collision_y = height
        self.time_since_last_move = 0.0
        self.pos = self._pick_new_target(width, self.collision_y)

    def _pick_new_target(self, width, height):
        return np.array([
            random.uniform(self.radius, width - self.radius),
            random.uniform(self.radius, height - self.radius)
        ], dtype=float)

    def update(self, dt, width, height):
        self.time_since_last_move += dt
        if self.time_since_last_move >= HOLD_DURATION:
            self.pos = self._pick_new_target(width, self.collision_y)
            self.time_since_last_move = 0.0

        # Clamp to valid region just in case
        self.pos[0] = max(self.radius, min(width - self.radius, self.pos[0]))
        self.pos[1] = max(self.radius, min(self.collision_y - self.radius, self.pos[1]))

    def draw(self, frame):
        y, x = np.ogrid[:frame.shape[0], :frame.shape[1]]
        dist = np.sqrt((x - self.pos[0])**2 + (y - self.pos[1])**2)
        mask = dist <= self.radius
        frame[mask] = [0, 255, 0, 255]
