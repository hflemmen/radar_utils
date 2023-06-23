import os

import cv2
import numpy as np
from typing import Tuple, List, Any


class VideoSink:

    def __init__(self, pathname: str, fps: int):
        """
        size = (height, width)
        """
        self.pathname = pathname
        self.fps = fps
        self.sink = None

    def add(self, img: np.ndarray):
        if self.sink is None:
            fourcc = cv2.VideoWriter_fourcc(*'FFV1')
            # fourcc = 0
            self.size = (img.shape[1], img.shape[0])
            path = self.pathname + '.avi'
            if os.path.exists(path):
                os.remove(path)
            os.makedirs('/'.join(path.split('/')[:-1]), exist_ok=True)
            self.sink = cv2.VideoWriter(path,
                                        apiPreference=cv2.CAP_FFMPEG,
                                        fourcc=fourcc, fps=self.fps, frameSize=self.size, isColor=True)

        t_img = None
        if img.shape[0] == self.size[1]:
            if img.shape[1] == self.size[0]:
                t_img = img
            elif img.shape[1] > self.size[0]:
                t_img = img[:, :self.size[0]]
            elif img.shape[1] < self.size[0]:
                t_img = np.hstack((img, np.zeros((self.size[1], self.size[0] - img.shape[1]), np.uint8)))
            else:
                raise ValueError(f"Wrong image dimensions. self.size:{self.size}, img.shape:{img.shape}")
        elif img.shape[0] > self.size[1]:
            if img.shape[1] == self.size[0]:
                t_img = img[:self.size[1], :]
            elif img.shape[1] > self.size[0]:
                t_img = img[:self.size[1], :self.size[0]]
            elif img.shape[1] < self.size[0]:
                t_img = np.hstack(
                    (img[:self.size[1], :], np.zeros((self.size[1], self.size[0] - img.shape[1]), dtype=np.uint8)))
            else:
                raise ValueError(f"Wrong image dimensions. self.size:{self.size}, img.shape:{img.shape}")
        elif img.shape[0] < self.size[1]:
            if img.shape[1] == self.size[0]:
                t_img = np.vstack((img, np.zeros((self.size[1] - img.shape[0], self.size[0], 3), dtype=np.uint8)))
            elif img.shape[1] > self.size[0]:
                t_img = np.vstack(
                    (img[:, :self.size[0]], np.zeros((self.size[1] - img.shape[0], self.size[0], 3), dtype=np.uint8)))
            elif img.shape[1] < self.size[0]:
                t_img = np.hstack(
                    (np.vstack(img, np.zeros((self.size[1] - img.shape[1], img.shape[1], 3), dtype=np.uint8)),
                     np.zeros((self.size[1], self.size[0] - img.shape[1], 3), dtype=np.uint8)))
            else:
                raise ValueError(f"Wrong image dimensions. self.size:{self.size}, img.shape:{img.shape}")
        else:
            raise ValueError(f"Wrong image dimensions. self.size:{self.size}, img.shape:{img.shape}")
        if (t_img.shape[1], t_img.shape[0]) == self.size:
            self.sink.write(t_img)
        else:
            raise ValueError(f"This error should be impossible to reach")

    def release(self):
        print("Saved video at:\t" + self.pathname + '.avi')
        self.sink.release()


class VideosSinkArray:

    def __init__(self, pathname: str, fps: int):
        self.sink = VideoSink(pathname, fps)

    def add(self, imgs: List[np.ndarray]):
        max_0 = max(map(lambda x: x.shape[0], imgs))
        # max_1 = max(map(lambda x: x.shape[1], imgs))

        res_img = np.concatenate(
            list(map(lambda x: cv2.copyMakeBorder(x, 0, max_0 - x.shape[0], 0, 0, cv2.BORDER_CONSTANT, 0), imgs)),
            axis=1)
        self.sink.add(res_img)
        # cv2.imshow("test", res_img)
        # cv2.waitKey(0)

    def release(self):
        self.sink.release()
