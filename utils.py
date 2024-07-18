# import the necessary packages
from pathlib import Path
import sys
import cv2
import depthai as dai
import numpy as np
import argparse
import json
import blobconverter
import glob
import os

def create_pipeline_images(nnPath):
    print("[INFO] initializing pipeline...")
    # initialize a depthai pipeline
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Provide config path for inference",
                        default='json/yolov4-tiny.json', type=str)
    args = parser.parse_args()

    # parse config
    configPath = Path(args.config)
    if not configPath.exists():
        raise ValueError("Path {} does not exist!".format(configPath))

    with configPath.open() as f:
        config = json.load(f)
    nnConfig = config.get("nn_config", {})

    # parse input shape
    if "input_size" in nnConfig:
        W, H = tuple(map(int, nnConfig.get("input_size").split('x')))

    # extract metadata
    metadata = nnConfig.get("NN_specific_metadata", {})
    classes = metadata.get("classes", {})
    coordinates = metadata.get("coordinates", {})
    anchors = metadata.get("anchors", {})
    anchorMasks = metadata.get("anchor_masks", {})
    iouThreshold = metadata.get("iou_threshold", {})
    confidenceThreshold = metadata.get("confidence_threshold", {})

    print(metadata)

    # parse labels
    nnMappings = config.get("mappings", {})
    labels = nnMappings.get("labels", {})
    
    if not Path(nnPath).exists():
        print("No blob found at {}. Looking into DepthAI model zoo.".format(nnPath))
        nnPath = str(blobconverter.from_zoo(args.model, shaves = 6, zoo_type = "depthai", use_cache=True))
    # sync outputs
    syncNN = True

    # Create pipeline
    pipeline = dai.Pipeline()

    # Define sources and outputs
    yoloIN = pipeline.createXLinkIn()
    yoloIN.setStreamName("in")
    detectionNetwork = pipeline.create(dai.node.YoloDetectionNetwork)
    detectionNetwork.setBlobPath(nnPath)
    yoloIN.out.link(detectionNetwork.input)
    nnOut = pipeline.createXLinkOut()
    nnOut.setStreamName("nn")
    detectionNetwork.out.link(nnOut.input)
    # Network specific settings
    detectionNetwork.setConfidenceThreshold(confidenceThreshold)
    detectionNetwork.setNumClasses(classes)
    detectionNetwork.setCoordinateSize(coordinates)
    detectionNetwork.setAnchors(anchors)
    detectionNetwork.setAnchorMasks(anchorMasks)
    detectionNetwork.setIouThreshold(iouThreshold)
    detectionNetwork.setNumInferenceThreads(2)
    detectionNetwork.input.setBlocking(False)
    
    # return the pipeline
    return pipeline

def softmax(x):
    # compute softmax values for each set of scores in x.
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def to_planar(arr: np.ndarray, shape: tuple) -> np.ndarray:
    # resize the image array and modify the channel dimensions
    resized = cv2.resize(arr, shape)
    return resized.transpose(2, 0, 1)

img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes
vid_formats = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes

class LoadImages:  # for inference
    def __init__(self, path, img_size=640, stride=32):
        p = str(Path(path).absolute())  # os-agnostic absolute path
        if '*' in p:
            files = sorted(glob.glob(p, recursive=True))  # glob
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
        elif os.path.isfile(p):
            files = [p]  # files
        else:
            raise Exception(f'ERROR: {p} does not exist')

        images = [x for x in files if x.split('.')[-1].lower() in img_formats]
        videos = [x for x in files if x.split('.')[-1].lower() in vid_formats]
        ni, nv = len(images), len(videos)

        self.img_size = img_size
        self.stride = stride
        self.files = images + videos
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv
        self.mode = 'image'
        if any(videos):
            self.new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nf > 0, f'No images or videos found in {p}. ' \
                            f'Supported formats are:\nimages: {img_formats}\nvideos: {vid_formats}'

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = 'video'
            ret_val, img0 = self.cap.read()
            if not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                else:
                    path = self.files[self.count]
                    self.new_video(path)
                    ret_val, img0 = self.cap.read()

            self.frame += 1
            # print(f'video {self.count + 1}/{self.nf} ({self.frame}/{self.nframes}) {path}: ', end='')

        else:
            # Read image
            self.count += 1
            img0 = cv2.imread(path)  # BGR
            assert img0 is not None, 'Image Not Found ' + path
            #print(f'image {self.count}/{self.nf} {path}: ', end='')

        return path, img0, self.cap

    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.nframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nf  # number of files

