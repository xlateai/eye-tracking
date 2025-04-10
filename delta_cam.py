import time
import numpy as np
import xospy
from utils import get_webcam_frame


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

        # Resize delta image to fit the screen
        cam_h, cam_w, _ = delta.shape
        start_y = height - cam_h
        start_x = (width - cam_w) // 2
        end_y = min(start_y + cam_h, height)
        end_x = min(start_x + cam_w, width)

        if 0 <= start_y < height and 0 <= start_x < width:
            frame[start_y:end_y, start_x:end_x, :3] = delta[:end_y - start_y, :end_x - start_x]
            frame[start_y:end_y, start_x:end_x, 3] = 255

        return frame


xospy.run_py_game(CamDeltaApp(), web=False, react_native=False)
