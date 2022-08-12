SS_CLASSES_LIST = ['sidewalk', 'broken_sidewalk', 'road', 'broken_road', 'crosswalk',
                   'speed_hump', 'braileblock', 'flowerbed', 'weed', 'sewer', 'manhole',
                   'stair', 'ramp', 'sidegap']

OD_CLASSES_LIST = ['barricade', 'beverage_desk', 'beverage_vending_machine', 'bicycle', 'bollard', 'bus', 'car',
                   'carrier', 'cat', 'chair', 'dog', 'door_normal', 'fire_hydrant', 'kiosk', 'lift', 'mailbox',
                   'motorcycle', 'movable_signage', 'parking_meter', 'person', 'pole', 'potted_plant',
                   'power_controller', 'resting_place_roof', 'scooter', 'stop', 'stroller', 'table', 'traffic_light',
                   'traffic_light_controller', 'traffic_sign', 'trash_can', 'tree_trunk', 'truck', 'wheelchair',
                   'unstable']


def get_class_name(num):
    return OD_CLASSES_LIST[num]


def get_cls_dict(mode):
    if mode == 'o':
        return OD_CLASSES_LIST
    elif mode == 's':
        return {i: n for i, n in enumerate(SS_CLASSES_LIST)}
