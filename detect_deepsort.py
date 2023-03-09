# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     path/                           # directory
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""
import glob

import argparse
import os
import platform
import sys
from pathlib import Path
import numpy as np
import torch.cuda
import json
import csv

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
import deep_sort


# DeepSORT -> Importing DeepSORT.
from deep_sort.deep_sort.detection import Detection
from deep_sort.application_util import preprocessing
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.tracker import Tracker
from deep_sort.tools import generate_detections as gdet
#from deep_sort.deep_sort_app import run as dp_run


from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode

# Counting -> Importing the counting class.
from counting import tracker_counter
from evaluation import evaluate_frame_count
from export_json import export_json, initialize_json

import cv2 as cv
import matplotlib.pyplot as plt

@smart_inference_mode()
def run(
        evaluate=True, #evaluate model and tracking performance
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='cuda device',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=0,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=True,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # DeepSORT -> Initializing tracker.
    max_cosine_distance = 0.5
    nn_budget = None
    nms_max_overlap = 1
    detections_out = []
    rows = []
    mot_export = []
    model_filename = 'deep_sort/resources/networks/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=32)
    metric = nn_matching.NearestNeighborDistanceMetric("euclidean", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    # Initialize counter.
    track_list = []
    track_list_id = []
    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    frame_count_total = 0
    frame_time_total = 0

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    frame_idx = 0
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        frame_idx = frame_idx + 1
        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # DeepSORT -> Extracting Bounding boxes and its confidence scores.
                    bboxes = []
                    scores = []
                    for *boxes, conf, cls in det:
                        bbox_left = min([boxes[0].item(), boxes[2].item()])
                        bbox_top = min([boxes[1].item(), boxes[3].item()])
                        bbox_w = abs(boxes[0].item() - boxes[2].item())
                        bbox_h = abs(boxes[1].item() - boxes[3].item())
                        box = [bbox_left, bbox_top, bbox_w, bbox_h]
                        bboxes.append(box)
                        scores.append(conf.item())
                        rows.append([frame_idx, -1, bbox_left, bbox_top, bbox_w, bbox_h, 1, 1, 1.0])

                    # DeepSORT -> Getting appearance features of the object.
                    features = encoder(im0, bboxes.copy())
                    # DeepSORT -> Storing all the required info in a list.
                    detections = [Detection(bbox, score, feature) for bbox, score, feature in
                                  zip(bboxes, scores, features)]

                    detections_out += [np.r_[(row, feature)] for row, feature in zip(rows, features)]
                    mot_export += [np.r_[row] for row in rows]
                    rows = []
                    # DeepSORT -> Predicting Tracks.
                    tracker.predict()
                    tracker.update(detections)
                    # track_time = time.time() - prev_time

                    # DeepSORT -> Plotting the tracks.
                    d = 0
                    y = 0
                    for track in tracker.tracks:
                        if not track.is_confirmed() or track.time_since_update > 1:
                            #print("Track is not confirmed or track time since update is greater than 1")
                            continue
                        if track.track_id not in track_list_id:
                            track_list_id.append(track.track_id)
                            track_list.append(tracker_counter(track, im0, p.name))

                            print("Found new track :", track.track_id)

                        if len(track.features) > 0:
                            print("Found a feature for track", track.track_id)
                        d += 1

                        # DeepSORT -> Changing track bbox to top left, bottom right coordinates.
                        bbox = list(track.to_tlbr())
                        while track.track_id != track_list[y].tracker_object.track_id:
                            y += 1
                            print(track_list[y].tracker_object.track_id)
                        track_list[y].count(frame_idx)
                        # DeepSORT -> Writing Track bounding box and ID on the frame using OpenCV.
                        txt = 'id:' + str(track.track_id) + ' fish count: ' + str(track_list[y].Fish_count)
                        (label_width, label_height), baseline = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
                        top_left = tuple(map(int, [int(bbox[0]), int(bbox[1]) - (label_height + baseline)]))
                        top_right = tuple(map(int, [int(bbox[0]) + label_width, int(bbox[1])]))
                        org = tuple(map(int, [int(bbox[0]), int(bbox[1]) - baseline]))

                        cv2.rectangle(im0, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 1)
                        cv2.rectangle(im0, top_left, top_right, (255, 0, 0), -1)
                        # Draw line
                        line_top_left = (int(track_list[y].line_as_box[0]), int(track_list[y].line_as_box[1]))
                        line_bottom_right = (int(track_list[y].line_as_box[2]), int(track_list[y].line_as_box[3]))
                        cv2.rectangle(im0, line_top_left, line_bottom_right, (255, 0, 0), 1)
                        # Draw center point
                        cv2.circle(im0, (int(track_list[y].bbox_center[0]), int(track_list[y].bbox_center[1])), 5, (255, 0, 0), 5)
                        cv2.putText(im0, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)

                        # DeepSORT -> Saving Track predictions into a text file.
                        save_format = '{frame},{id},{x1},{y1},{w},{h},{x},{y},{z}\n'
                        with open(txt_path + '.txt', 'a') as f:
                            line = save_format.format(frame=frame_idx, id=track.track_id, x1=int(bbox[0]),
                                                      y1=int(bbox[1]), w=int(bbox[2] - bbox[0]),
                                                      h=int(bbox[3] - bbox[1]), x=-1, y=-1, z=-1)
                            f.write(line)
                        y += 1

            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")
        frame_time_total += dt[1].dt * 1E3
        frame_count_total += 1
    output_detections_path = str(save_dir / p.name).replace('.mp4', '.npy')
    output_mot_path = str(save_dir / p.name).replace('.mp4', '_det.txt')
    #output_filename = r"D:\OneDrive\Github\yolov5\runs\test\fish_video0\fish_video0_custom.npy"
    #output_filename_mot = r"D:\OneDrive\Github\yolov5\runs\test\fish_video0\det.txt"
    np.save(
        output_detections_path, np.asarray(detections_out), allow_pickle=False)
    np.savetxt(output_mot_path, np.asarray(mot_export), fmt='%d', delimiter=',')

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if evaluate:
        sum = 0
        exit_frame = []
        enter_frame = []
        fish_count_frames_list = []
        for track in track_list:
            for frame, count_seq in zip(track.Exit_frame, track.fish_count_frames):
                exit_frame.append(frame)
                fish_count_frames_list.append(count_seq)
            for frame in track.Enter_frame:
                enter_frame.append(frame)
            sum += track.Fish_count
        # TODO: find the right order for appending the fish_count_frames, since we are dealing with multiple tracks.
        # Order the fish_count_frames by the frame number
        comb_list = zip(exit_frame, fish_count_frames_list)
        sorted_list = sorted(comb_list)
        fish_count_frames_list_sorted = [i[1] for i in sorted_list]
        export_json('results.json', p.name, exit_frame, enter_frame, sum, fish_count_frames_list_sorted)
        print(10 * '-' + ' Total count: ' + str(sum) + ' ' + 10 * '-')
        print('Total time: ', frame_time_total)
        print('Total frames: ', frame_count_total)
        print('Average time per frame: ', frame_time_total / frame_count_total)
        print('Average FPS: ', frame_count_total / (frame_time_total / 1000))

        # Save Total time and Total frames to a csv file
        with open('total_time_frames.csv', 'a+', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([p.name, frame_time_total, frame_count_total])




    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluate', action='store_true', default=True, help='evaluate model and tracking performance')
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5x.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.50, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.50, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt, files):
    check_requirements(exclude=('tensorboard', 'thop'))
    if opt.evaluate:
        initialize_json('results.json')
    if len(files) > 0:
        for f in files:
            opt.source = f
            run(**vars(opt))
    else:
        run(**vars(opt))
    if opt.evaluate:
        evaluate_frame_count('results.json')



if __name__ == "__main__":
    opt = parse_opt()
    try:
        files = glob.glob(opt.source)  # check valid path
    except:
        files = []

    main(opt, files)

