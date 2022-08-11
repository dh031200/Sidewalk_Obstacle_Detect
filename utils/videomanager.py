import os
import cv2
import numpy as np


class VideoManager:
    def __init__(self, src, dst, record):
        p = dst.split('/')
        os.makedirs('/'.join(p[:-1]), exist_ok=True)
        self.cap = cv2.VideoCapture(src)
        self.dst = dst
        self.color_changer = True
        self.record = record

        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.framecount = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.shape = int(self.width * self.height)
        if record:
            self.writer = self.init_writer()

    # def set_depth_input_size(self, depth_model_input_size):
    #     self.depth_width = depth_model_input_size[0]
    #     self.depth_height = depth_model_input_size[1]

    def read(self, cc=False):
        _, frame = self.cap.read()
        if frame is None:
            return None
        return frame

    def pre_processing(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.depth_width, self.depth_height))
        return img

    def init_writer(self):
        writer = cv2.VideoWriter(self.dst, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (self.width, self.height))
        return writer

    def post_processing(self, img):
        img *= 255
        img = cv2.resize(img, (self.width, self.height))
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = np.uint8(img)
        return img

    def resize_and_concat(self, img_1, img_2, img_3, img_4):
        img = cv2.vconcat([
            cv2.hconcat([cv2.resize(img_1, (self.width // 2, self.height // 2)),
                         cv2.resize(img_2, (self.width // 2, self.height // 2))]),
            cv2.hconcat([cv2.resize(img_3, (self.width // 2, self.height // 2)),
                         cv2.resize(img_4, (self.width // 2, self.height // 2))])])
        if self.record:
            self.write(img)
        return img

    def write(self, img):
        self.writer.write(img)

    def release(self):
        self.cap.release()
        if self.record:
            self.writer.release()
