import time
import numpy as np
import xospy
from utils import get_webcam_frame
import cv2


class CamDeltaApp(xospy.ApplicationBase):
    def setup(self, state):
        xospy.video.webcam.init_camera()
        self.last_frame = get_webcam_frame()
        self.last_time = time.time()

    def tick(self, state):
        now = time.time()
        dt = now - self.last_time
        self.last_time = now

        width, height = state.frame.width, state.frame.height
        mv = memoryview(state.frame.buffer)
        frame = np.frombuffer(mv, dtype=np.uint8).reshape((height, width, 4))
        frame[:] = 0

        current_frame = get_webcam_frame()
        delta = current_frame.astype(np.int16) - self.last_frame.astype(np.int16)
        self.last_frame = current_frame

        # Normalize delta to 0–255 and clip
        delta = np.clip((delta + 128), 0, 255).astype(np.uint8)

        # Resize delta frame to fill the screen
        delta_resized = cv2.resize(delta, (width, height), interpolation=cv2.INTER_LINEAR)
        delta_resized = cv2.cvtColor(delta_resized, cv2.COLOR_GRAY2RGB)

        # Fill RGB channels
        frame[:, :, :3] = delta_resized
        frame[:, :, 3] = 255  # Full alpha

        return frame




xospy.run_py_game(CamDeltaApp(), web=False, react_native=False)
