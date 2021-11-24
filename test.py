import argparse
import time
from pathlib import Path
import os
import glob
import cv2
import torch
import numpy as np
from utils.general import check_img_size, non_max_suppression, scale_coords,\
    xyxy2xywh, strip_optimizer
from utils.torch_utils import select_device, time_synchronized
from utils.google_utils import attempt_download
from utils.plots import plot_one_box
import json
from numpy import random


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True,
              scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)
    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]
    dw /= 2  # divide padding into 2 sides
    dh /= 2
    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=color)
    return img, ratio, (dw, dh)


class LoadImages:
    def __init__(self, path, img_size=320):
        p = str(Path(path))
        p = os.path.abspath(p)
        if '*' in p:
            files = sorted(glob.glob(p, recursive=True))
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '*.*')))
        elif os.path.isfile(p):
            files = [p]
        else:
            raise Exception('ERROR: %s does not exist' % p)
        img_formats = ['png']
        images = [x for x in files if x.split('.')[-1].lower() in img_formats]
        self.ni = len(images)
        self.img_size = img_size
        self.files = images
        self.mode = 'images'

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.ni:
            raise StopIteration
        path = self.files[self.count]
        # Read image
        self.count += 1
        img0 = cv2.imread(path)
        print('image %g/%g %s: ' % (self.count, self.ni, path), end='')
        img = letterbox(img0, new_shape=self.img_size)[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        return path, img, img0

    def __len__(self):
        return self.ni


def detect(save_img=False):
    source, weights, save_txt, imgsz = opt.source, opt.weights,
    opt.save_txt, opt.img_size

    # Directories
    save_dir = Path(str(Path(opt.project)))
    (save_dir if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    # Initialize
    device = select_device(opt.device)
    half = device.type != 'cpu'

    # Load model
    attempt_download(weights[0])
    model = torch.load(weights[0], map_location=device)['model']
    model.float().fuse().eval()
    imgsz = check_img_size(imgsz, s=model.stride.max())
    if half:
        model.half()

    # Set Dataloader
    dataset = LoadImages(source, img_size=imgsz)
    save_img = True

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None

    for path, img, im0s in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        torch.cuda.synchronize()
        t1 = time.time()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres,
                                   classes=opt.classes,
                                   agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):
            p, s, im0 = Path(path), '', im0s
            save_path = str(save_dir / p.name)
            txt_path = str(save_dir / p.stem)
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4],
                                          im0.shape).round()
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:
                        xywh_1 = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))/gn)
                        b = xywh_1.view(-1).tolist()
                        line = (cls, *b, conf) if opt.save_conf else (cls, *b)
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
                    if save_img:
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label,
                                     color=colors[int(cls)], line_thickness=3)

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))
            if save_img:
                cv2.imwrite(save_path, im0)

    # Combine to answer.json
    filepath = str(save_dir)+"/"
    data = []

    def key_func(x):
        number = int(os.path.basename(x).replace(".png", ""))
        return number
    file_name = sorted(glob.glob(filepath + "*.png"), key=key_func)
    print(file_name)

    for i, file in enumerate(file_name):
        img_name = file
        b_name = os.path.basename(img_name)
        if not os.path.isfile(file.replace(".png", ".txt")):
            a = {"image_id": int(b_name.replace(".png", "")),
                 "score": float(0.5),
                 "category_id": int(0),
                 "bbox": [1, 1, 1, 1]}
            data.append(a)
        else:
            # load txt and png
            f = open(file.replace(".png", ".txt"), 'r')
            contents = f.readlines()
            im = cv2.imread(img_name)
            h, w, c = im.shape

            for content in contents:
                a = {"image_id": [], "score": [],
                     "category_id": [], "bbox": []}
                content = content.replace('\n', '')
                c = content.split(' ')

                a["category_id"] = int(c[0])
                w_center = w*float(c[1])
                h_center = h*float(c[2])
                width = w*float(c[3])
                height = h*float(c[4])
                left = w_center - width/2
                top = h_center - height/2

                a["bbox"] = [left, top, width, height]
                a["score"] = float(c[5])
                bb_name = os.path.basename(img_name)
                a["image_id"] = int(bb_name.replace(".png", ""))
                data.append(a)
                f.close()

    ret = json.dumps(data,
                     separators=(',', ': '), ensure_ascii=False, indent=4)
    with open('answer.json', 'w') as fp:
        fp.write(ret)
    print('{} data save to answer.json'.format(len(data)))
    print('Done. (%.3fs)' % (time.time() - t0))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5m.pt')
    parser.add_argument('--source', type=str, default='data/images')
    parser.add_argument('--img-size', type=int, default=640)
    parser.add_argument('--conf-thres', type=float, default=0.25)
    parser.add_argument('--iou-thres', type=float, default=0.45)
    parser.add_argument('--device', default='0')
    parser.add_argument('--save-txt', action='store_true')
    parser.add_argument('--save-image', action='store_true')
    parser.add_argument('--save-conf', action='store_true')
    parser.add_argument('--classes', nargs='+', type=int)
    parser.add_argument('--agnostic-nms', action='store_true')
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--update', action='store_true')
    parser.add_argument('--project', default='detect')
    parser.add_argument('--name', default='exp')
    parser.add_argument('--exist-ok', action='store_true')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        if opt.update:
            for opt.weights in ['yolov5m.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
