import argparse
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
# from strong_sort.utils.parser import get_config
# from strong_sort.strong_sort import StrongSORT
from tracker.byte_tracker import BYTETracker


cudnn.benchmark = True


def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size

    save_img = not opt.nosave and not source.endswith('.txt')

    # Read config
    config = Config()
    models = Model(config, dual_gpu=True, img_size=imgsz)
    device = models.device_1

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    tracker = BYTETracker(opt)
    # cfg = get_config(opt.config_strongsort)
    # cfg.merge_from_file()

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

    # strongsort = StrongSORT(
    #     strong_sort_weights,
    #     device,
    #     half,
    #     max_dist=cfg.STRONGSORT.MAX_DIST,
    #     max_iou_distance=cfg.STRONGSORT.MAX_IOU_DISTANCE,
    #     max_age=cfg.STRONGSORT.MAX_AGE,
    #     n_init=cfg.STRONGSORT.N_INIT,
    #     nn_budget=cfg.STRONGSORT.NN_BUDGET,
    #     mc_lambda=cfg.STRONGSORT.MC_LAMBDA,
    #     ema_alpha=cfg.STRONGSORT.EMA_ALPHA,
    # )

    # For Debug
    t0 = time.time()
    cnt = 0
    end = 1500

    # Loop start
    cf, pf = None, None
    for path, img, im0s, vid_cap in tqdm(dataset):
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
        # ss_img = models.seg_draw(im0s,ss_result)
        ss_img = ss_vis.draw_surface(surface_tracker.canvas, surface_tracker.disappear, surface_tracker.disappear_limit)
        im0s = cv2.addWeighted(im0s, 1, ss_img, 0.5, 0)
        im0s = ss_vis.draw_legends(im0s)
        # im0s[ss_img != (0,0,0)] = ss_img[ss_img != (0,0,0)]
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
        # de_pred = np.repeat(1 - de_pred[:, :, np.newaxis], 3, axis=2)
        # de_pred = cv2.resize(de_pred, (1920, 1080), interpolation=cv2.INTER_LANCZOS4).astype(np.float64)
        # de_pred /= 255.0
        # de_pred[de_pred < 0.87] *= 0.3

        # Process detections
        # im0s = im0s.astype(np.float64) * de_pred
        # im0s
        # print(od_pred)
        # print('----------------------')
        # print('----------------------')
        # print(od_pred)
        # print('----------------------')
        # print(online_targets)
        # exit()
        for i, det in enumerate(od_pred):  # detections per image
            im0 = im0s
            # cf = im0 = im0s
            # if cfg.STRONGSORT.ECC:  # camera motion compensation
            #     strongsort.tracker.camera_update(prev_frames[i], curr_frames[i])
            # print('-------------------------------------------')
            # print(i, det)
            # print('-------------------------------------------')
            # det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            # print(i, det)
            # print('-------------------------------------------')
            # print(reversed(det))
            # exit()

            # im0 = im0s.astype(np.uint8)
            # im0 = cv2.resize(cv2.applyColorMap(de_pred, cv2.COLORMAP_JET), (1920,1080),interpolation=cv2.INTER_LANCZOS4)
            # im0 = cv2.resize(cv2.cvtColor(de_pred,cv2.COLOR_GRAY2BGR), (1920,1080),interpolation=cv2.INTER_LANCZOS4)
            # im0 = cv2.resize(de_pred, (1920,1080),interpolation=cv2.INTER_LANCZOS4)

            p = Path(path)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                online_targets = tracker.update(det, [dataset.height, dataset.width], [dataset.height, dataset.width])
                # Write results
                for target in online_targets:
                    label = f'{target.track_id}_{names[target.clss]} {target.score:.2f}'
                    plot_one_box(target.tlbr, im0, label=label, color=colors[int(target.clss)], line_thickness=3)
                # for *xyxy, conf, cls in reversed(det):
                #     label = f'{names[int(cls)]} {conf:.2f}'
                #
                #     # plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                #     plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                #     # print(f'xyxy: {list(map(int, xyxy))}, conf: {conf}, cls: {cls}')
                #     print(f'xyxy: {xyxy}, conf: {conf}, cls: {cls}')
                # # print('-----------------------')
                # # print(online_targets)
                # # print('-----------------------')
                # for i in online_targets:
                #     print(f'{i}  clss : {i.clss}. score : {i.score}')
                #     print(*map(int, i.tlbr))
                # print('-----------------------')
            # For show streaming
            # cv2.imshow('frame', im0)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

            # Save results (image with detections)
            if save_img:
                # if dataset.mode == 'image':
                #     cv2.imwrite(save_path, im0)
                #     print(f" The image with the result is saved in: {save_path}")
                # else:  # 'video' or 'stream'
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

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    # parser.add_argument('--config-strongsort', type=str, default='strong_sort/configs/strong_sort.yaml')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
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
    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6,
                        help="threshold for filtering out boxes of which aspect ratio are above the given value.")
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    opt = parser.parse_args()
    print(opt, '\n')

    with torch.no_grad():
        detect()
