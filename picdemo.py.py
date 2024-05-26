#!/usr/bin/python
# -*- coding: UTF-8 -*-
import time
import argparse
import os

import cv2
import torch
from torchvision.transforms import Compose

from networks.transforms import Resize
from networks.transforms import PrepareForNet

def process_depth(dep):
    dep = dep - dep.min()
    dep = dep / dep.max()
    dep_vis = dep * 255

    return dep_vis.astype('uint8')

def load_image_path(args):
    image_path = args.input
    return [image_path], [os.path.basename(image_path)]

def run(args):
    print("Initialize")

    # select device
    device = torch.device("cpu")
    print("Device: %s" % device)

    # load network
    print("Creating model...")
    if args.model == 'large':
        from networks import MidasNet
        model = MidasNet(args)
    else:
        from networks import TCSmallNet
        model = TCSmallNet(args)

    if os.path.isfile(args.resume):
        model.load_state_dict(torch.load(args.resume, map_location='cpu'))
        print("Loading model from " + args.resume)
    else:
        print("Loading model path fail, model path does not exist.")
        exit()

    model.cpu().eval()
    print("Loading model done...")

    transform = Compose([
        Resize(
            args.resize_size,  # width
            args.resize_size,  # height
            resize_target=None,
            keep_aspect_ratio=True,
            ensure_multiple_of=32,
            resize_method="lower_bound",
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        PrepareForNet(),
    ])

    # get input
    path_list, scene_names = load_image_path(args)

    # prepare output folder
    os.makedirs(args.output, exist_ok=True)

    start_time = time.time()  # Record the start time

    for i in range(len(path_list)):
        print("Processing: %s" % scene_names[i])
        img = cv2.imread(path_list[i])

        # predict depth
        output_list = []
        with torch.no_grad():
            frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frame = transform({"image": frame})["image"]
            frame = torch.from_numpy(frame).to(device).unsqueeze(0)

            prediction = model.forward(frame)
            print(prediction.min(), prediction.max())
            prediction = (torch.nn.functional.interpolate(
                prediction,
                size=img.shape[:2],
                mode="bilinear",
                align_corners=False,
            ).squeeze().cpu().numpy())
            output_list.append(prediction)

            end_time = time.time()  # Record the end time

        # save output
        output_name = os.path.join(args.output, scene_names[i] + '.png')
        output_list = [process_depth(out) for out in output_list]

        cv2.imwrite(output_name, output_list[0])


    execution_time = end_time - start_time
    print(args.output + " Done.")
    print("Execution time: {:.2f} seconds".format(execution_time))

if __name__ == "__main__":
    # set torch options
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # Settings
    parser = argparse.ArgumentParser(description="A PyTorch Implementation of Single Image Depth Estimation")

    parser.add_argument('--model', default='large', choices=['small', 'large'], help='size of the model')
    parser.add_argument('--resume', type=str, required=True, help='path to checkpoint file')
    parser.add_argument('--input', default='./input/image.jpg', type=str, help='input image path')
    parser.add_argument('--output', default='./output', type=str, help='path to save output')
    parser.add_argument('--resize_size',
                        type=int,
                        default=384,
                        help="spatial dimension to resize input (default: small model:256, large model:384)")

    args = parser.parse_args()

    print("Run Single Image Depth Estimation")
    run(args)
