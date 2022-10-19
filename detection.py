import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadWebcam
from utils.general import check_img_size, non_max_suppression, scale_coords, set_logging
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized

print(torch.cuda.is_available())


class Detection:
    def __init__(self):
        # Press 'q' to quit video
        # Write direction to your pt weights
        self.weights = 'your_weights.pt'
        # Write direction to your video
        self.source = 'your_video.mp4'

    def detect(self):
        set_logging()
        if torch.cuda.is_available() == True:
            device = select_device('0')
        else:
            device = select_device('cpu')

        half = device.type != 'cpu'

        model = attempt_load(self.weights, map_location=device)
        stride = int(model.stride.max())
        imgsz = check_img_size(640, s=stride)

        if half:
            model.half()

        cudnn.benchmark = True
        dataset = LoadWebcam(self.source, img_size=imgsz, stride=stride)

        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))

        for img, im0s in dataset:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()
            img /= 255.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            t1 = time_synchronized()
            with torch.no_grad():
                pred = model(img)[0]
            t2 = time_synchronized()

            pred = non_max_suppression(pred, 0.25, 0.45)
            t3 = time_synchronized()

            for i, det in enumerate(pred):

                s, im0, frame = '%g: ' % i, im0s[i].copy(), dataset.count

                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()

                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()
                    print(n)
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                for *xyxy, conf, cls in reversed(det):
                    label = f'{names[int(cls)]} {conf:.2f}'
                    im0s = plot_one_box(xyxy, im0s, label=label, color=colors[int(cls)], line_thickness=1)
                    print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

                cv2.imshow('Test', im0s)


dt = Detection()
dt.detect()
