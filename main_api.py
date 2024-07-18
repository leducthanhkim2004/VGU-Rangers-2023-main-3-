#!/usr/bin/env python3
from pathlib import Path
import sys
import cv2
import depthai as dai
import numpy as np
import time
import argparse
import json
import blobconverter
import csv

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", help="Provide model name or model path for inference",
                    default='yolov4_tiny_coco_416x416', type=str)
parser.add_argument("-c", "--config", help="Provide config path for inference",
                    default='/home/pi/VGU-Rangers-2023-main/last_tiny.json', type=str)
parser.add_argument("-r", "--record", help="Record the video during running model. Provide video name with .avi format", type=str, default='')
args = parser.parse_args()

if args.record:
    assert args.record.endswith('.avi'), 'Incorrect video format.'

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

# parse labels
nnMappings = config.get("mappings", {})
labels = nnMappings.get("labels", {})

# get model path
nnPath = args.model
if not Path(nnPath).exists():
    nnPath = str(blobconverter.from_zoo(args.model, shaves=6, zoo_type="depthai", use_cache=True))

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
camRgb = pipeline.create(dai.node.ColorCamera)
detectionNetwork = pipeline.create(dai.node.YoloDetectionNetwork)
xoutRgb = pipeline.create(dai.node.XLinkOut)
nnOut = pipeline.create(dai.node.XLinkOut)

xoutRgb.setStreamName("rgb")
nnOut.setStreamName("nn")

# Properties
camRgb.setPreviewSize(W, H)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
camRgb.setFps(10)

# Network specific settings
detectionNetwork.setConfidenceThreshold(confidenceThreshold)
detectionNetwork.setNumClasses(classes)
detectionNetwork.setCoordinateSize(coordinates)
detectionNetwork.setAnchors(anchors)
detectionNetwork.setAnchorMasks(anchorMasks)
detectionNetwork.setIouThreshold(iouThreshold)
detectionNetwork.setBlobPath(nnPath)
detectionNetwork.setNumInferenceThreads(2)
detectionNetwork.input.setBlocking(False)

# Linking
camRgb.preview.link(detectionNetwork.input)
detectionNetwork.passthrough.link(xoutRgb.input)
detectionNetwork.out.link(nnOut.input)

# Set up a CSV file
file_name = 'demo.csv'
csv_file = open(file_name, 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Label', 'Confidence', 'X_min', 'Y_min', 'X_max', 'Y_max'])

# Connect to device and start pipeline
with dai.Device(pipeline) as device:
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    qDet = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

    frame = None
    detections = []
    startTime = time.monotonic()
    counter = 0
    color2 = (255, 255, 255)

    object_counts = {}

    # nn data normalization
    def frameNorm(frame, bbox):
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

    # Frame display and CSV writing
    def displayFrame(name, frame, detections, video):
        for detection in detections:
            bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
            label = labels[detection.label]
            confidence = f"{int(detection.confidence * 100)}%"
            csv_writer.writerow([label, confidence, *bbox])
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
            cv2.putText(frame, label, (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(frame, confidence, (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)

            if label in object_counts:
                object_counts[label] += 1
            else:
                object_counts[label] = 1

        if video:
            video.write(frame)
        cv2.imshow(name, frame)

    video = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*'MJPG'), 10, (W, H)) if args.record else None

    # Flag to check if 15 seconds have passed
    fifteen_seconds_passed = False

    while True:
        inRgb = qRgb.get()
        inDet = qDet.get()

        if inRgb:
            frame = inRgb.getCvFrame()
            cv2.putText(frame, "NN fps: {:.2f}".format(counter / (time.monotonic() - startTime)),
                        (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color2)

        if inDet:
            detections = inDet.detections
            counter += 1

        if frame is not None:
            displayFrame("rgb", frame, detections, video)

        # Check if 15 seconds have passed
        elapsed_time = time.monotonic() - startTime
        if elapsed_time > 15 and not fifteen_seconds_passed:
            print("Object counts after 15 seconds:")
            for label, count in object_counts.items():
                print(f"{label}: {count}")
            print(f"Elapsed time: {elapsed_time:.2f} seconds")
            fifteen_seconds_passed = True

        if cv2.waitKey(1) == ord('q'):
            break

    if video:
        video.release()

    # Write object counts to CSV
    csv_writer.writerow(['Label', 'Count'])
    for label, count in object_counts.items():
        csv_writer.writerow([label, count])

    csv_file.close()

    print("Detection results have been saved to demo.csv")
