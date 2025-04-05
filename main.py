import xospy
import numpy as np
import time
import math
import random
from PIL import Image, ImageDraw, ImageFont
import torch


RENDER_VIDEO = False
VELOCITY = 512


def get_webcam_frame() -> np.ndarray:
    cam_w, _ = xospy.video.webcam.get_resolution()
    cam_bytes = xospy.video.webcam.get_frame()
    bytes_per_pixel = 3
    total_pixels = len(cam_bytes) // bytes_per_pixel
    cam_h = total_pixels // cam_w

    if cam_w * cam_h * bytes_per_pixel != len(cam_bytes):
        raise Exception("Webcam resolution doesn't match buffer size. Skipping.")

    cam_array = np.frombuffer(cam_bytes, dtype=np.uint8).reshape((cam_h, cam_w, 3))

    # Downscale to height 256 with aspect ratio preserved
    scale = cam_h / 256
    new_w = int(cam_w / scale)
    new_h = 256
    cam_array = np.array(Image.fromarray(cam_array).resize((new_w, new_h), Image.LANCZOS))

    cam_array = cam_array[:, ::-1]
    
    # Convert to grayscale (single channel)
    cam_array = np.mean(cam_array, axis=2).astype(np.uint8)
    cam_array = np.expand_dims(cam_array, axis=2)


    return cam_array


def draw_cross(frame: np.ndarray, x: float, y: float, size: int = 10, color=(255, 0, 0, 255)):
    height, width, _ = frame.shape
    x, y = int(x), int(y)

    for dx in range(-size, size + 1):
        xi = x + dx
        if 0 <= xi < width and 0 <= y < height:
            frame[y, xi] = color

    for dy in range(-size, size + 1):
        yi = y + dy
        if 0 <= x < width and 0 <= yi < height:
            frame[yi, x] = color


def draw_loss_text_on_ball(frame: np.ndarray, loss: float, x: float, y: float):
    """Draw the current loss inside the ball in small black text."""
    h, w, _ = frame.shape
    img = Image.fromarray(frame, mode='RGBA')
    draw = ImageDraw.Draw(img)

    text = f"{loss:.4f}"
    try:
        font = ImageFont.truetype("Arial.ttf", 12)
    except:
        font = ImageFont.load_default()

    text_w, text_h = draw.textsize(text, font=font)
    text_x = int(x - text_w / 2)
    text_y = int(y - text_h / 2)

    text_x = max(0, min(w - text_w, text_x))
    text_y = max(0, min(h - text_h, text_y))

    draw.text((text_x, text_y), text, fill=(0, 0, 0, 255), font=font)
    frame[:] = np.array(img)


class EyeTracker(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, 1, kernel_size=16, stride=4),
            torch.nn.Sigmoid(),
            torch.nn.Conv2d(1, 1, kernel_size=7, stride=2),
            torch.nn.Sigmoid(),
            torch.nn.Conv2d(1, 1, kernel_size=3, stride=1),
            torch.nn.Sigmoid(),
            torch.nn.Conv2d(1, 1, kernel_size=3, stride=1),
            torch.nn.Sigmoid(),
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d((3, 3)),
            torch.nn.Flatten(),
            torch.nn.Linear(9, 2),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.decoder(x)
        return x


class Ball:
    def __init__(self, width, height):
        self.pos = np.array([width / 2, height / 2], dtype=float)
        self.angle = random.uniform(0, 2 * math.pi)
        self.radius = 30 * 0.85
        self.elapsed_time = 0.0
        self.target_angle = self._pick_new_angle()
        self.angle_lerp_speed = 0.5

    def _pick_new_angle(self):
        return random.uniform(0, 2 * math.pi)

    def update(self, dt, width, height):
        self.elapsed_time += dt
        velocity_mod = 1.0 + 0.3 * math.sin(self.elapsed_time * 0.5)
        current_speed = VELOCITY * velocity_mod

        angle_diff = (self.target_angle - self.angle + math.pi) % (2 * math.pi) - math.pi
        max_angle_step = self.angle_lerp_speed * dt
        if abs(angle_diff) < max_angle_step:
            self.angle = self.target_angle
            self.target_angle = self._pick_new_angle()
        else:
            self.angle += max(-max_angle_step, min(angle_diff, max_angle_step))

        delta = np.array([math.cos(self.angle), math.sin(self.angle)]) * current_speed * dt
        self.pos += delta

        camera_reserved_height = int(height * 0.10)
        bottom_limit = height - camera_reserved_height

        if self.pos[0] - self.radius < 0:
            self.pos[0] = self.radius
            self.angle = math.pi - self.angle
        if self.pos[0] + self.radius > width:
            self.pos[0] = width - self.radius
            self.angle = math.pi - self.angle
        if self.pos[1] - self.radius < 0:
            self.pos[1] = self.radius
            self.angle = -self.angle
        if self.pos[1] + self.radius > bottom_limit:
            self.pos[1] = bottom_limit - self.radius
            self.angle = -self.angle

        self.angle %= 2 * math.pi

    def draw(self, frame):
        y, x = np.ogrid[:frame.shape[0], :frame.shape[1]]
        dist = np.sqrt((x - self.pos[0])**2 + (y - self.pos[1])**2)
        mask = dist <= self.radius
        frame[mask] = [0, 255, 0, 255]  # neon green


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
        self.training_enabled = True  # starts as True

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
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.step_count += 1

            # draw_loss_text_on_ball(frame, loss.item(), self.ball.pos[0], self.ball.pos[1])

        pred_x = math.floor(float(pred[0, 0].item()) * width)
        pred_y = math.floor(float(pred[0, 1].item()) * height)

        if self.training_enabled:
            print(f"[{self.step_count}] loss: {loss.item():.6f} / px={pred_x}, py={pred_y}")
        else:
            print(f"px={pred_x}, py={pred_y}")
        
        draw_cross(frame, pred_x, pred_y)

        # --- Draw training toggle message ---
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

        # --- Render webcam feed in bottom zone ---
        cam_h, cam_w, _ = cam_frame.shape
        start_y = height - cam_h
        start_x = (width - cam_w) // 2

        if 0 <= start_y < height and 0 <= start_x < width:
            end_y = min(start_y + cam_h, height)
            end_x = min(start_x + cam_w, width)
            frame[start_y:end_y, start_x:end_x, :3] = cam_frame[:end_y-start_y, :end_x-start_x]
            frame[start_y:end_y, start_x:end_x, 3] = 255

        return frame



xospy.run_py_game(PyApp(), web=False, react_native=False)
