import numpy as np
import cv2
from PIL import ImageFont, ImageDraw, Image

FONTPATH = "utils/MaruBuri-Regular.ttf"
FONT = ImageFont.truetype(FONTPATH, 30)
ALPHA = 0.6
# FONT = cv2.FONT_HERSHEY_PLAIN
TEXT_SCALE = 1.3
TEXT_THICKNESS = 1


def gen_colors(num_colors):
    import random
    import colorsys
    num_colors -= 1
    hsvs = [[float(x) / num_colors, 1., 0.7] for x in range(num_colors)]
    random.seed(1234)
    random.shuffle(hsvs)
    rgbs = list(map(lambda x: list(colorsys.hsv_to_rgb(*x)), hsvs))
    bgrs = [(int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
            for rgb in rgbs] + [(0, 0, 255)]
    return bgrs


def gen_surface_colors():
    return [
        (0, 255, 0),  # sidewalk
        (0, 140, 255),  # broken_sidewalk
        (255, 0, 0,),  # road
        (160, 0, 255),  # broken_road
        (180, 180, 180),  # crosswalk
        (40, 230, 185),  # speed_hump
        (120, 190, 240),  # braileblock
        (230, 60, 200),  # flowerbed
        (130, 200, 130),  # weed
        (10, 130, 180),  # sewer
        (30, 140, 90),  # manhole
        (150, 120, 90),  # stair
        (130, 70, 180),  # ramp
        (70, 70, 70),  # sidegap
    ]


def draw_boxed_text(img, text, topleft, color):
    assert img.dtype == np.uint8
    img_h, img_w, _ = img.shape
    if topleft[0] >= img_w or topleft[1] >= img_h:
        return img
    margin = 4
    size = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, TEXT_SCALE, TEXT_THICKNESS)

    w = int(size[0][0] * 1.3)
    h = size[0][1] + margin * 4
    patch = np.zeros((h, w, 3), dtype=np.uint8)
    patch[...] = color
    img_pil = Image.fromarray(patch)
    draw = ImageDraw.Draw(img_pil)
    draw.text((3, 0), text, font=FONT, fill=(255, 255, 255, 0))
    patch = np.array(img_pil)

    if topleft[1] < 20:
        tl = topleft[1]
    else:
        tl = topleft[1] + 20

    w = min(w, img_w - topleft[0])
    h = min(h, img_h - tl)
    roi = img[tl:tl + h, topleft[0]:topleft[0] + w, :]
    cv2.addWeighted(patch[0:h, 0:w, :], ALPHA, roi, 1 - ALPHA, 0, roi)
    return img


class Visualizer:
    def __init__(self, width, height, cls_dict):
        self.width = width
        self.height = height
        self.cls_dict = cls_dict
        self.colors = gen_surface_colors()

    def draw_bboxes(self, img, boxes, clss, confs, disappear):
        for i in boxes:
            if disappear[i] > 0:
                continue
            x_min, y_min, x_max, y_max = boxes[i]
            color = self.colors[clss[i]]
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 7)
            txt_loc = (max(x_min + 2, 0), max(y_min - 18, 0))
            cls_name = self.cls_dict.get(clss[i], 'CLS{}'.format(clss[i]))
            txt = '{} {:.1f}%'.format(cls_name, confs[i] * 100)
            img = draw_boxed_text(img, txt, txt_loc, color)
            cv2.putText(img, str(i), (x_min, y_min), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

        return img

    def draw_seg(self, segs, clss, conf, disappear):
        img = np.full((self.height, self.width, 3), (0, 0, 0), dtype=np.uint8)
        for i in segs:
            if disappear[i] > 3 or conf[i] < 0.3:
                continue
            color = self.colors[clss[i]]
            for x, y in np.argwhere(segs[i]):
                img[x, y] = color
        return img

    def draw_legends(self, img):
        cv2.rectangle(img, (self.width - 280, 10), (self.width - 10, 310), (255, 255, 255), -1)
        for i, color in enumerate(gen_surface_colors()):
            cv2.putText(img, self.cls_dict[i], (self.width - 270, 35 + (i * 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color)
            cv2.line(img, (self.width - 70, 30 + (i * 20)), (self.width - 20, 30 + (i * 20)), color, 10)
        return img

    def draw_surface(self, canvas, disappear, limit):
        img = np.full((self.height, self.width, 3), (0, 0, 0), dtype=np.uint8)
        for i in range(len(self.colors)):
            img[np.where(canvas == i)] = self.colors[i]
        return img

    # def distance(self, val, max_v, size):
    #     dt = np.linspace(1, 10, max_v + 1)
    #     distance = (13.5 - np.log1p(size)) / np.sqrt(np.log1p(size)) * (dt[val] ** 2) / (np.log1p(size) * 4)
    #     return distance
