# ball_pathing.py
import numpy as np
import random
import math

VELOCITY = 256


class Ball:
    def __init__(self, width, height):
        self.pos = np.array([width / 2, height / 2], dtype=float)
        self.radius = 30 * 0.85
        self.speed = VELOCITY
        self.collision_y = height
        self.target = self._pick_new_target(width, self.collision_y)

    def _pick_new_target(self, width, height):
        return np.array([
            random.uniform(self.radius, width - self.radius),
            random.uniform(self.radius, height - self.radius)
        ], dtype=float)

    def update(self, dt, width, height):
        if self.target[1] > self.collision_y or np.linalg.norm(self.target - self.pos) < self.speed * dt:
            self.target = self._pick_new_target(width, self.collision_y)

        direction = self.target - self.pos
        direction /= np.linalg.norm(direction)
        self.pos += direction * self.speed * dt

        self.pos[0] = max(self.radius, min(width - self.radius, self.pos[0]))
        self.pos[1] = max(self.radius, min(self.collision_y - self.radius, self.pos[1]))

    def draw(self, frame):
        y, x = np.ogrid[:frame.shape[0], :frame.shape[1]]
        dist = np.sqrt((x - self.pos[0])**2 + (y - self.pos[1])**2)
        mask = dist <= self.radius
        frame[mask] = [0, 255, 0, 255]
