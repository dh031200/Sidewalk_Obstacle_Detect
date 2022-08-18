import argparse
import os
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from tqdm import tqdm

from models.model import Model
from utils.datasets import LoadImages
from utils.general import non_max_suppression, scale_coords, set_logging, increment_path, xyxy2xywh
from utils.plots import plot_one_box
from utils.classlist import get_cls_dict
from utils.config import Config
from utils.tracker import Surface_tracker
from utils.visualization import Visualizer, gen_colors
from tracker.byte_tracker import BYTETracker, StableTracker


cudnn.benchmark = True



def detect(opt):
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    # source, weights, view_img, imgsz = opt.source, opt.weights, opt.view_img, opt.img_size

    depth_box_padding = 3

    save_img = not opt.nosave and not source.endswith('.txt')

    # Read config
    config = Config()
    models = Model(config, dual_gpu=True, img_size=imgsz)
    device = models.device_1

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    os.makedirs(f'{save_dir}/img/', exist_ok=True)

    print(f'save_dir : {save_dir}')

    # Initialize
    set_logging()
    tracker = BYTETracker(opt)

    stable_tracker = StableTracker()


    # Set Dataloader
    vid_path, vid_writer = None, None
    dataset = LoadImages(source, img_size=models.img_size, stride=models.stride)

    # Get names and colors
    names = get_cls_dict('o')
    colors = gen_colors(len(names))

    # Gen tracker
    surface_tracker = Surface_tracker(dataset.width, dataset.height, disappear_limit=2)
    ss_vis = Visualizer(dataset.width, dataset.height, get_cls_dict('s'))
    models.od_model(torch.zeros(1, 3, models.img_size, models.img_size).to(device).type_as(
        next(models.od_model.parameters())))  # run once
    old_img_w = old_img_h = models.img_size
    old_img_b = 1

    # For Debug
    t0 = time.time()
    cnt = 0
    end = 1500

    # Loop start
    cf, pf = None, None

    for path, img, img0s, vid_cap in tqdm(dataset):
        im0s = img0s.copy()
        # cnt += 1
        # if cnt > end:
        #     break
        img = torch.from_numpy(img).to(device)
        img = img.half()
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        ss_result = models.seg_inference(im0s)
        surface_tracker.update(ss_result)
        ss_img = ss_vis.draw_surface(surface_tracker.canvas, surface_tracker.disappear, surface_tracker.disappear_limit)
        im0s = cv2.addWeighted(im0s, 1, ss_img, 0.5, 0)
        im0s = ss_vis.draw_legends(im0s)

        # Warmup
        if device.type != 'cpu' and (
                old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                _ = models.od_model(img, augment=opt.augment)[0]

        # OD Inference
        od_pred = models.od_model(img, augment=opt.augment)[0]

        # Apply NMS
        od_pred = non_max_suppression(od_pred, opt.conf_thres, opt.iou_thres, classes=opt.classes,
                                      agnostic=opt.agnostic_nms)

        # DE Inference
        de_pred = models.de_model(img)
        de_pred = de_pred['pred_d'].squeeze().cpu().numpy()
        de_pred = (de_pred / de_pred.max()) * 255
        de_pred = de_pred.astype(np.uint8)

        for i, det in enumerate(od_pred):  # detections per image
            im0 = im0s

            p = Path(path)  # to Path

            save_path = str(save_dir / p.name)  # img.jpg
            # print(f'save_path : {save_path}')

            if len(det):
                print()
                depth = torch.zeros(det.size()[0], 1).to(device)
                for idx, bbox in enumerate(det[:, :4]):
                    x_min, y_min, x_max, y_max = map(int, bbox)
                    # depth[idx] = de_pred[(y_min + y_max) // 2, (x_min + x_max) // 2]
                    depth[idx] = de_pred[y_min + depth_box_padding:y_max - depth_box_padding,
                                 x_min + depth_box_padding:x_max - depth_box_padding].mean()

                # Rescale boxes from img_size to im0 size
                det = torch.cat([det, depth], dim=1)
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                online_targets = tracker.update(det, [dataset.height, dataset.width], [dataset.height, dataset.width])
                # stable_tracker_status = stable_tracker.update(
                #     [target for target in online_targets if np.bincount(target.clss_pool).argmax() != 35])
                # Write results
                online_to_stable = []
                for target in online_targets:
                    if np.bincount(target.clss_pool).argmax() != 35:
                        online_to_stable.append(target)
                    # label = f'{target.track_id}_{names[target.clss]} {target.score:.2f}'

                    # label = f'{target.track_id}_{names[target.clss]} {target.depth}'
                    # plot_one_box(target.tlbr, im0, label=label, color=colors[int(target.clss)], line_thickness=3)
                    print(
                        f'id:{target}, bbox: {target.tlbr}, cls: {names[target.clss]},'
                        f' depth: {target.depth:>3.2f}, score: {target.score:>2.4f},'
                        f' clss_pool: {target.clss_pool} n: {target.n}')
                    # print(f'stable tracker : {stable_tracker}')
                print('----------------------------------------------------')
                stable_tracker_status = stable_tracker.update(online_to_stable)
                for target in stable_tracker_status:
                    label = f'{target.id}_{names[target.clss]} {target.depth:>6.2f}'
                    plot_one_box(target.tlbr, im0, label=label, color=colors[int(target.clss)], line_thickness=3)
                    print(f'id: {target}, bbox: {target.tlbr}, cls: {names[target.clss]:^24s}, '
                          f'depth: {target.depth:>6.2f}, score: {target.score:>6.3f}, disappear: {target.disappear}')
            print('===============================================================================')
            # For show streaming
            # cv2.imshow('frame', im0)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

            # Save results (image with detections)
            if save_img:

                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path += '.mp4'
                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer.write(im0)
                disappeared_track = [t for t in stable_tracker.stable_stracks if
                                     t.is_alive and t.appeared >= 100 and not t.img_save]
                for t in disappeared_track:
                    t.img_save = True
                    xmin, ymin, xmax, ymax = map(int, t.tlbr)
                    cv2.imwrite(f'{save_dir}/img/{str(t.id).zfill(4)}_{names[int(t.clss)]}.png',
                                img0s[max(0, ymin):min(img0s.shape[0] - 1, ymax),
                                      max(0, xmin):min(img0s.shape[1] - 1, xmax)])
    print(f'Done. ({time.time() - t0:.3f}s)')

    return stable_tracker


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.35, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.55, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument("--track_thresh", type=float, default=0.75, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=75, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.90, help="matching threshold for tracking")
    parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6,
                        help="threshold for filtering out boxes of which aspect ratio are above the given value.")
    parser.add_argument('--min_box_area', type=float, default=80, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    args = parser.parse_args()
    print(args, '\n')

    with torch.no_grad():
        od_result = detect(args)

    # for s in od_result.stable_stracks:
    #     if s.appeared > 150:
    #         dict(clss=s.clss, s.)
