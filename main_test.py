# import the necessary packages
import utils
import os
import numpy as np
import cv2
import depthai as dai

nnPath = r"C:\Users\PC\OneDrive\Desktop\Tech\VGU\Tum_2023\VGU-Rangers-2023\last_tiny_openvino_2022.1_6shave.blob"

# initialize a depthai images pipeline
print("[INFO] initializing a depthai image pipeline...")
pipeline = utils.create_pipeline_images(nnPath)

TEST_DATA = r"C:\Users\PC\OneDrive\Desktop\Tech\VGU\Tum_2023\TestData\test_img"
IMG_DIM = (640, 640)
LABELS = ["bicycle",
            "motorcycle",
            "car",
            "bus",
            "truck"]
OUTPUT_DIR = r"C:\Users\PC\OneDrive\Desktop\Tech\VGU\Tum_2023\TestData\result"

def frameNorm(frame, bbox):
    normVals = np.full(len(bbox), frame.shape[0])
    normVals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

def displayFrame(frame, detections):
    color = (255, 0, 0)
    for detection in detections:
        bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
        cv2.putText(frame, LABELS[detection.label], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
        cv2.putText(frame, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

# pipeline defined, now the device is assigned and pipeline is started
with dai.Device(pipeline) as device:

    device.setLogLevel(dai.LogLevel.DEBUG)
    device.setLogOutputLevel(dai.LogLevel.DEBUG)
    usbSpeed = device.getUsbSpeed()
    print(usbSpeed)
    cmx_usage = device.getCmxMemoryUsage()
    print(cmx_usage.remaining, cmx_usage.used)
    
    
    # define the queues that will be used in order to communicate with
    # depthai and then send our input image for predictions
    yoloIN = device.getInputQueue("in")
    qDet = device.getOutputQueue(name="nn", maxSize=4, blocking=False)
    
    print("[INFO] loading image from disk...")
    dataset = utils.LoadImages(TEST_DATA, img_size=640)
    
    vid_writer = None
    vid_path = None
    
    for i, (path, image0, vid_cap) in enumerate(dataset):
        # load the input image and then resize it
        image = image0.copy()
        nn_data = dai.NNData()
        nn_data.setLayer(
            "input",
            utils.to_planar(image, IMG_DIM)
        )
        yoloIN.send(nn_data)
        
        print("[INFO] fetching neural network output for {}".
            format(os.path.basename(path)))
        
        inDet = qDet.get()
        
        # apply softmax on predictions and
        # fetch class label and confidence score
        if inDet is not None:
            detections = inDet.detections
            for detection in detections:
                displayFrame(image, detections)
        
        save_path = os.path.join(OUTPUT_DIR, os.path.basename(path))
        
        if dataset.video_flag[i]:
            if vid_path != save_path:  # new video
                vid_path = save_path
                if isinstance(vid_writer, cv2.VideoWriter):
                    vid_writer.release()  # release previous video writer
                if vid_cap:  # video
                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                else:  # stream
                    fps, w, h = 30, image0.shape[1], image0.shape[0]
                    save_path += '.mp4'
                vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            vid_writer.write(image)
        else:
            cv2.imwrite(
            save_path,
            image
            )
