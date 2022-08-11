import sys
from collections import OrderedDict

import torch
from models.glpdepth import GLPDepth
from mmdet.apis import init_detector, inference_detector
from models.experimental import attempt_load
from utils.torch_utils import select_device, time_synchronized, TracedModel
from utils.general import check_img_size


class Model:
    def __init__(self, config, dual_gpu, img_size):
        self.stride = None
        self.img_size = None

        print('CORE: Check cuda available')
        self.device_1, self.device_2, self.device_3 = cuda_checker(dual_gpu)

        print('CORE: Load object detection model')
        try:
            self.od_model = self.init_od_model(config.od_attribute['checkpoint_path'], img_size)
        except:
            print('ERROR: Failed!')
            sys.exit(-1)
        print('** Success! **\n')

        print('CORE: Load surface segmentation model')
        try:
            self.ss_model = self.init_ss_model(config.ss_attribute['config_path'],
                                               config.ss_attribute['checkpoint_path'])
        except:
            print('ERROR: Failed!')
            sys.exit(-1)
        print('** Success! **\n')

        print('CORE: Load depth estimation model')
        try:
            self.de_model = self.init_depth_model(config.depth_attribute['checkpoint_path'])
        except:
            print('ERROR: Failed!')
            sys.exit(-1)
        print('** Success! **\n')

    def init_od_model(self, od_cp, img_size, trace=True, half=True):
        print('load checkpoint from local path:', od_cp)
        device = self.device_1
        model = attempt_load(od_cp, map_location=device)
        stride = int(model.stride.max())
        img_size = check_img_size(img_size, s=stride)
        if trace:
            model = TracedModel(model, device, img_size)
        if half:
            model.half()
        self.stride = stride
        self.img_size = img_size
        return model

    def init_ss_model(self, ss_cfg, ss_cp):
        return init_detector(ss_cfg, ss_cp, device=self.device_2)

    def init_depth_model(self, weight_path):
        print('load checkpoint from local path:', weight_path)
        model = GLPDepth(max_depth=10.0, is_train=False).to(self.device_3)
        model_weight = torch.load(weight_path)
        if 'module' in next(iter(model_weight.items()))[0]:
            model_weight = OrderedDict((k[7:], v) for k, v in model_weight.items())
        model.load_state_dict(model_weight)
        model.half()
        model.eval()
        return model

    def detect_inference(self, img):
        return inference_detector(self.od_model, img)

    def seg_inference(self, img):
        return inference_detector(self.ss_model, img)

    def detection_draw(self, img, od_result):
        return self.od_model.show_result(img, od_result)

    def seg_draw(self, img, ss_result):
        return self.ss_model.show_result(img, ss_result)


def cuda_checker(dual_gpu):
    if torch.cuda.is_available():
        if dual_gpu:
            device_1, device_2, device_3 = 'cuda:1', 'cuda:0', 'cuda:1'
            print('** Using cuda : ', torch.cuda.get_device_name(0), ', ', torch.cuda.get_device_name(1), '**')
        else:
            device_1 = device_2 = device_3 = 'cuda:0'
            print('** Using cuda : ', torch.cuda.get_device_name(0), '**')
    else:
        device_1 = device_2 = device_3 = 'cpu'
        print('** Using cpu **')
    print()

    return device_1, device_2, device_3
