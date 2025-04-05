# main.py
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import xospy
from utils import get_webcam_frame, draw_cross
from ball_pathing import Ball


class EfficientEyeTracker(nn.Module):
    def __init__(self, h, w):
        super().__init__()
        self.h = h
        self.w = w

        self.attention = nn.Parameter(torch.ones(h, w))
        self.row_mu = nn.Parameter(torch.tensor(h / 2.0))
        self.row_logstd = nn.Parameter(torch.zeros(1))
        self.col_mu = nn.Parameter(torch.tensor(w / 2.0))
        self.col_logstd = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        x = x.squeeze(0).squeeze(0)  # (h, w)
        weighted = x * self.attention

        row_sum = weighted.mean(dim=1)  # (h,)
        col_sum = weighted.mean(dim=0)  # (w,)

        # Sample from Normal distributions using REINFORCE
        row_std = torch.exp(self.row_logstd)
        col_std = torch.exp(self.col_logstd)

        row_dist = torch.distributions.Normal(self.row_mu, row_std)
        col_dist = torch.distributions.Normal(self.col_mu, col_std)

        row_sample = row_dist.rsample()  # allow backprop through sample
        col_sample = col_dist.rsample()

        # Clamp and normalize
        row_sample = row_sample.clamp(0, self.h - 1) / self.h
        col_sample = col_sample.clamp(0, self.w - 1) / self.w

        output = torch.stack([col_sample, row_sample])  # (2,)
        return output.unsqueeze(0), row_dist, col_dist


class PyApp(xospy.ApplicationBase):
    def setup(self, state):
        xospy.video.webcam.init_camera()
        self.last_time = time.time()
        self.tick_count = 0
        self.ball = Ball(state.frame.width, state.frame.height)

        cam_height, cam_width = get_webcam_frame().shape[:2]
        self.model = EfficientEyeTracker(cam_height, cam_width)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)
        self.step_count = 0
        self.training_enabled = True

    def on_mouse_down(self, state):
        self.training_enabled = not self.training_enabled
        print("Training enabled:", self.training_enabled)

    def tick(self, state):
        self.optimizer.zero_grad()
        self.tick_count += 1
        now = time.time()
        dt = now - self.last_time
        self.last_time = now

        width, height = state.frame.width, state.frame.height
        mv = memoryview(state.frame.buffer)
        frame = np.frombuffer(mv, dtype=np.uint8).reshape((height, width, 4))
        frame[:] = 0

        cam_frame = get_webcam_frame()
        cam_h, cam_w, _ = cam_frame.shape

        collision_y = height - cam_h
        self.ball.collision_y = collision_y

        if self.training_enabled:
            self.ball.update(dt, width, height)
            self.ball.draw(frame)

        x = torch.from_numpy(cam_frame).permute(2, 0, 1).unsqueeze(0).float() / 250
        pred, row_dist, col_dist = self.model(x)

        target_x = torch.tensor([self.ball.pos[0] / width], dtype=torch.float32)
        target_y = torch.tensor([self.ball.pos[1] / height], dtype=torch.float32)
        target = torch.stack([target_x, target_y]).unsqueeze(0)

        if self.training_enabled:
            reward = -F.mse_loss(pred, target.detach())  # higher reward = better

            log_prob = row_dist.log_prob(pred[0, 1] * self.model.h) + \
                       col_dist.log_prob(pred[0, 0] * self.model.w)

            loss = -log_prob * reward  # REINFORCE objective
            loss.backward()
            self.optimizer.step()
            self.step_count += 1

        pred_x = math.floor(float(pred[0, 0].item()) * width)
        pred_y = min(math.floor(float(pred[0, 1].item()) * height), collision_y - 1)

        if self.training_enabled:
            print(f"[{self.step_count}] REINFORCE loss: {loss.item():.6f} / px={pred_x}(tx={int(self.ball.pos[0])}), py={pred_y}(ty={int(self.ball.pos[1])})")
        else:
            print(f"px={pred_x}, py={pred_y}")

        draw_cross(frame, pred_x, pred_y)

        try:
            pil_img = Image.fromarray(frame, mode='RGBA')
            draw = ImageDraw.Draw(pil_img)
            try:
                font_size = 48
                font = ImageFont.truetype("Arial.ttf", font_size)
            except:
                font = ImageFont.load_default()

            text = "Click to pause training" if self.training_enabled else "Click to resume training"
            draw.text((30, height - font_size * 2), text, font=font, fill=(255, 255, 255, 255))
            frame[:] = np.array(pil_img)
        except Exception as e:
            print("Failed to draw training message:", e)

        start_y = height - cam_h
        start_x = (width - cam_w) // 2

        if 0 <= start_y < height and 0 <= start_x < width:
            end_y = min(start_y + cam_h, height)
            end_x = min(start_x + cam_w, width)
            frame[start_y:end_y, start_x:end_x, :3] = cam_frame[:end_y-start_y, :end_x-start_x]
            frame[start_y:end_y, start_x:end_x, 3] = 255

        frame[collision_y:collision_y+2, :, :] = [0, 255, 0, 255]

        return frame


xospy.run_py_game(PyApp(), web=False, react_native=False)
