# main.py
import time
import math
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import xospy
from utils import get_webcam_frame, draw_cross
from ball_pathing import Ball


class EyeTracker(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, 4, kernel_size=32, stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(4, 8, kernel_size=16, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 4, kernel_size=8, stride=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(4, 1, kernel_size=7, stride=1),
            torch.nn.ReLU(),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d((8, 8)),
            torch.nn.Flatten(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 2),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.decoder(x)
        return x


model = EyeTracker()


class PyApp(xospy.ApplicationBase):
    def setup(self, state):
        xospy.video.webcam.init_camera()
        self.last_time = time.time()
        self.tick_count = 0
        self.ball = Ball(state.frame.width, state.frame.height)

        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.loss_fn = torch.nn.MSELoss()
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

        x = torch.from_numpy(cam_frame).permute(2, 0, 1).unsqueeze(0).float() / 100
        pred = self.model(x)

        if self.training_enabled:
            target_x = torch.tensor([self.ball.pos[0] / width], dtype=torch.float32)
            target_y = torch.tensor([self.ball.pos[1] / height], dtype=torch.float32)
            target = torch.stack([target_x, target_y]).unsqueeze(0)

            loss = self.loss_fn(pred, target)
            loss.backward()
            self.optimizer.step()

            self.step_count += 1

        pred_x = math.floor(float(pred[0, 0].item()) * width)
        pred_y = min(math.floor(float(pred[0, 1].item()) * height), collision_y - 1)

        if self.training_enabled:
            print(f"[{self.step_count}] loss: {loss.item():.6f} / px={pred_x}, py={pred_y}")
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