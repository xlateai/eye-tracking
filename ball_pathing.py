# ball_pathing.py
import numpy as np
import random

HOLD_DURATION = 5.0  # seconds


class Ball:
    def __init__(self, width, height):
        self.original_radius = 64
        self.collision_y = height
        self.time_since_last_move = 0.0
        self.pos = self._pick_new_target(width, self.collision_y)

    def _pick_new_target(self, width, height):
        return np.array([
            random.uniform(self.original_radius, width - self.original_radius),
            random.uniform(self.original_radius, height - self.original_radius)
        ], dtype=float)

    def update(self, dt, width, height):
        self.time_since_last_move += dt
        if self.time_since_last_move >= HOLD_DURATION:
            self.pos = self._pick_new_target(width, self.collision_y)
            self.time_since_last_move = 0.0

        self.pos[0] = max(self.original_radius, min(width - self.original_radius, self.pos[0]))
        self.pos[1] = max(self.original_radius, min(self.collision_y - self.original_radius, self.pos[1]))

    def draw(self, frame):
        t = min(self.time_since_last_move / HOLD_DURATION, 1.0)
        current_radius = self.original_radius * (1.0 - t)

        if current_radius <= 0:
            return  # Nothing to draw

        y, x = np.ogrid[:frame.shape[0], :frame.shape[1]]
        dist = np.sqrt((x - self.pos[0])**2 + (y - self.pos[1])**2)
        mask = dist <= current_radius
        frame[mask] = [0, 255, 0, 255]
