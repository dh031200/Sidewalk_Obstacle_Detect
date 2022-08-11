import json

class Config:
    def __init__(self):
        with open('config.json', "r") as config:
            config_info = json.load(config)
        
        # self.od_attribute = config_info["m2f_model"]
        self.od_attribute = config_info["yolo"]
        self.ss_attribute = config_info["surface_segmentation_model_n2xt"]
        self.depth_attribute = config_info["depth_estimation_model"]
        self.core_attribute = config_info["core"]
