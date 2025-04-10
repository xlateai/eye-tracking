import time
import math
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import xospy
from utils import get_webcam_frame, draw_cross
from ball_pathing import Ball
from model import EfficientEyeTracker


class PyApp(xospy.ApplicationBase):
    def setup(self, state):
        xospy.video.webcam.init_camera()
        self.last_time = time.time()
        self.tick_count = 0
        self.ball = Ball(state.frame.width, state.frame.height)

        cam_height, cam_width = get_webcam_frame().shape[:2]
        self.model = EfficientEyeTracker(cam_height, cam_width)

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

        x = torch.from_numpy(cam_frame).permute(2, 0, 1).float() / 250.0
        x = x.unsqueeze(0)  # (1, 3, H, W)

        pred = self.model(x)  # (1, 2)
        pred_x, pred_y = pred[0, 0], pred[0, 1]

        loss = None
        if self.training_enabled:
            target_xy = torch.tensor([[self.ball.pos[0] / width, self.ball.pos[1] / height]], dtype=torch.float32)
            loss, _ = self.model.update(x, target_xy)
            self.step_count += 1

        px = math.floor(float(pred_x.item()) * width)
        py = math.floor(float(pred_y.item()) * height)

        if self.training_enabled:
            print(f"[{self.step_count}] loss: {loss:.6f} / px={px}(tx={int(self.ball.pos[0])}), "
                  f"py={py}(ty={int(self.ball.pos[1])})")
        else:
            print(f"px={px}, py={py}")

        # Overlay text
        # try:
        #     pil_img = Image.fromarray(frame, mode='RGBA')
        #     draw = ImageDraw.Draw(pil_img)
        #     try:
        #         font = ImageFont.truetype("Arial.ttf", 48)
        #     except:
        #         font = ImageFont.load_default()
        #     text = "Click to pause training" if self.training_enabled else "Click to resume training"
        #     text_width, text_height = draw.textsize(text, font=font)
        #     text_x = (width - text_width) // 2
        #     draw.text((text_x, 20), text, font=font, fill=(255, 255, 255, 255))
        #     frame[:] = np.array(pil_img)
        # except Exception as e:
        #     print("Failed to draw training message:", e)

        # Draw webcam + attention overlays
        start_y = height - cam_h
        start_x = (width - cam_w) // 2
        end_y = min(start_y + cam_h, height)
        end_x = min(start_x + cam_w, width)

        if 0 <= start_y < height and 0 <= start_x < width:
            frame[start_y:end_y, start_x:end_x, :3] = cam_frame[:end_y - start_y, :end_x - start_x]
            frame[start_y:end_y, start_x:end_x, 3] = 255

            # att_img = attention_to_image(self.model.attention)
            # h, w = att_img.shape[:2]
            # h = min(h, end_y - start_y)
            # w_half = min(w // 2, start_x, width - end_x)

            # frame[start_y:start_y + h, 0:w_half] = att_img[:h, :w_half]
            # frame[start_y:start_y + h, end_x:end_x + w_half] = att_img[:h, -w_half:]

        frame[collision_y:collision_y + 2, :, :] = [0, 255, 0, 255]

        if self.training_enabled:
            self.ball.draw(frame)

        draw_cross(frame, px, py)
        return frame


xospy.run_py_game(PyApp(), web=False, react_native=False)
