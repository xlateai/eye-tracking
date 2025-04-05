# main.py
import time
import math
import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import xospy
from utils import get_webcam_frame, draw_cross
from ball_pathing import Ball


def attention_to_image(attention: torch.Tensor) -> np.ndarray:
    """
    Converts a 2D attention tensor to a grayscale RGBA image array.

    Args:
        attention (Tensor): (h, w) attention weights

    Returns:
        np.ndarray: (h, w, 4) RGBA image
    """
    att = attention.detach().cpu().numpy()
    att = (att - att.min()) / (att.ptp() + 1e-6)  # Normalize to 0–1
    att_img = (att * 255).astype(np.uint8)
    rgba = np.stack([att_img] * 3 + [np.full_like(att_img, 255)], axis=-1)
    return rgba


class EfficientEyeTracker(nn.Module):
    def __init__(self, h, w):
        super().__init__()
        self.attention = nn.Parameter(torch.ones(h, w))
        self.row_weights = nn.Parameter(torch.ones(h))
        self.col_weights = nn.Parameter(torch.ones(w))

    def forward(self, x):
        gray = x.mean(dim=1, keepdim=True)  # Convert to grayscale
        weighted = gray.squeeze(1) * self.attention  # (batch, h, w)

        row_sum = weighted.mean(dim=2)  # (batch, h)
        col_sum = weighted.mean(dim=1)  # (batch, w)

        row_output = (row_sum * self.row_weights).mean(dim=1)  # (batch,)
        col_output = (col_sum * self.col_weights).mean(dim=1)  # (batch,)

        output = torch.stack([col_output, row_output], dim=1)  # (batch, 2)
        return torch.sigmoid(output)


class PyApp(xospy.ApplicationBase):
    def setup(self, state):
        xospy.video.webcam.init_camera()
        self.last_time = time.time()
        self.tick_count = 0
        self.ball = Ball(state.frame.width, state.frame.height)

        cam_height, cam_width = get_webcam_frame().shape[:2]

        self.model_x = EfficientEyeTracker(cam_height, cam_width)
        self.model_y = EfficientEyeTracker(cam_height, cam_width)

        self.optimizer_x = torch.optim.Adam(self.model_x.parameters(), lr=0.1)
        self.optimizer_y = torch.optim.Adam(self.model_y.parameters(), lr=0.1)

        self.loss_fn = torch.nn.MSELoss()
        self.step_count = 0
        self.training_enabled = True

    def on_mouse_down(self, state):
        self.training_enabled = not self.training_enabled
        print("Training enabled:", self.training_enabled)

    def tick(self, state):
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

        x = torch.from_numpy(cam_frame).permute(2, 0, 1).float() / 250.0
        x = x.unsqueeze(0)  # Add batch dim: (1, 3, h, w)

        pred_x = self.model_x(x)[:, 0]
        pred_y = self.model_y(x)[:, 1]

        if self.training_enabled:
            target_x = torch.tensor([self.ball.pos[0] / width], dtype=torch.float32)
            target_y = torch.tensor([self.ball.pos[1] / height], dtype=torch.float32)

            self.optimizer_x.zero_grad()
            self.optimizer_y.zero_grad()

            loss_x = self.loss_fn(pred_x.squeeze(), target_x)
            loss_y = self.loss_fn(pred_y.squeeze(), target_y)

            loss_x.backward()
            loss_y.backward()

            self.optimizer_x.step()
            self.optimizer_y.step()

            self.step_count += 1

        px = math.floor(float(pred_x.item()) * width)
        py = min(math.floor(float(pred_y.item()) * height), collision_y - 1)

        if self.training_enabled:
            print(f"[{self.step_count}] loss_x: {loss_x.item():.6f} / px={px}(tx={int(self.ball.pos[0])}), "
                  f"loss_y: {loss_y.item():.6f} / py={py}(ty={int(self.ball.pos[1])})")
        else:
            print(f"px={px}, py={py}")

        draw_cross(frame, px, py)

        # Draw overlay text at top-center
        try:
            pil_img = Image.fromarray(frame, mode='RGBA')
            draw = ImageDraw.Draw(pil_img)
            try:
                font_size = 48
                font = ImageFont.truetype("Arial.ttf", font_size)
            except:
                font = ImageFont.load_default()

            text = "Click to pause training" if self.training_enabled else "Click to resume training"
            text_width, text_height = draw.textsize(text, font=font)
            text_x = (width - text_width) // 2
            text_y = 20
            draw.text((text_x, text_y), text, font=font, fill=(255, 255, 255, 255))
            frame[:] = np.array(pil_img)
        except Exception as e:
            print("Failed to draw training message:", e)

        # Draw webcam and attention overlays
        start_y = height - cam_h
        start_x = (width - cam_w) // 2
        end_y = min(start_y + cam_h, height)
        end_x = min(start_x + cam_w, width)

        if 0 <= start_y < height and 0 <= start_x < width:
            # Webcam feed
            frame[start_y:end_y, start_x:end_x, :3] = cam_frame[:end_y - start_y, :end_x - start_x]
            frame[start_y:end_y, start_x:end_x, 3] = 255

            # Left: x attention
            att_img_x = attention_to_image(self.model_x.attention)
            h, w = att_img_x.shape[:2]
            h = min(h, end_y - start_y)
            w = min(w, start_x)
            frame[start_y:start_y + h, 0:w] = att_img_x[:h, :w]

            # Right: y attention
            att_img_y = attention_to_image(self.model_y.attention)
            h, w = att_img_y.shape[:2]
            w = min(w, width - end_x)
            h = min(h, end_y - start_y)
            frame[start_y:start_y + h, end_x:end_x + w] = att_img_y[:h, :w]

        frame[collision_y:collision_y + 2, :, :] = [0, 255, 0, 255]

        return frame


xospy.run_py_game(PyApp(), web=False, react_native=False)
