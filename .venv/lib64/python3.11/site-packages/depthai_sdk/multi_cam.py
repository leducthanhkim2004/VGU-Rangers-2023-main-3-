from typing import Dict, Any, Optional, List, Union, Callable
import depthai as dai
from pathlib import Path
import threading
import logging
import time

from depthai_sdk.oak_camera import OakCamera
from depthai_sdk.record import RecordType, Record

from depthai_sdk.components.camera_component import CameraComponent
from depthai_sdk.components.component import Component
from depthai_sdk.components.imu_component import IMUComponent
from depthai_sdk.components.nn_component import NNComponent
from depthai_sdk.components.stereo_component import StereoComponent
from depthai_sdk.components.pointcloud_component import PointcloudComponent
from depthai_sdk.visualize.visualizer import Visualizer

from depthai_sdk.classes.packet_handlers import (
    BasePacketHandler,
    QueuePacketHandler,
    RosPacketHandler,
    TriggerActionPacketHandler,
    RecordPacketHandler,
    CallbackPacketHandler,
    VisualizePacketHandler
)

def flatten(lst):
    flat_list = []
    for item in lst:
        if isinstance(item, list):
            flat_list.extend(flatten(item))
        else:
            flat_list.append(item)
    return flat_list

class OakCameras:
    def __init__(self,
                 devices: Optional[list] = None,
                 usb_speed: Union[None, str, dai.UsbSpeed] = None,  # Auto by default
                 rotation: int = 0,
                 args: Union[bool, Dict] = True):

        if devices is None:
            device_infos = dai.Device.getAllAvailableDevices()
            logging.debug(f'Found {len(device_infos)} devices')
            devices = [device_info.getMxId() for device_info in device_infos]

        if len(devices) == 0:
            raise RuntimeError('No OAK devices found!')

        self.devices: List[OakCamera] = []

        threads = []
        for device in devices:
            # If user already passed OakCamera
            if type(device) is OakCamera:
                self.devices.append(device)
                continue

            def connect(device: str, devices: List[OakCamera]):
                try:
                    logging.info(f'Connecting to {device}')
                    devices.append(OakCamera(device, usb_speed=usb_speed, rotation=rotation, args=args))
                except Exception as e:
                    logging.error(f'Failed to connect to {device}: {e}')

            time.sleep(1) # Currently required due to XLink race issues
            thread = threading.Thread(target=connect, args=(device, self.devices))
            thread.start()
            threads.append(thread)

        for t in threads:
            t.join() # Wait for all threads to finish

        if len(self.devices) == 0:
            raise RuntimeError('No OAK devices connected!')

        self._packet_handlers = []
        self._polling = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, tb):
        for device in self.devices:
            device.close()

    def camera(self,
            source: Union[str, dai.CameraBoardSocket],
            resolution: Optional[Union[
                str, dai.ColorCameraProperties.SensorResolution, dai.MonoCameraProperties.SensorResolution
                ]] = None,
            fps: Optional[float] = None,
            encode: Union[None, str, bool, dai.VideoEncoderProperties.Profile] = None,
            name: Optional[str] = None,
            ) -> List[CameraComponent]:
        return [device.camera(source, resolution, fps, encode, name) for device in self.devices]

    def all_cameras(self,
                resolution: Optional[Union[
                    str, dai.ColorCameraProperties.SensorResolution, dai.MonoCameraProperties.SensorResolution
                    ]] = None,
                fps: Optional[float] = None,
                encode: Union[None, str, bool, dai.VideoEncoderProperties.Profile] = None,
                ) -> List[CameraComponent]:
        cams = []
        for device in self.devices:
            cams.extend(device.all_cameras(resolution, fps, encode))
        return cams

    def create_nn(self,
                  model: Union[str, Dict, Path],
                  inputs: List[Union[CameraComponent, NNComponent]],
                  nn_type: Optional[str] = None,
                  tracker: bool = False,  # Enable object tracker - only for Object detection models
                  spatials: Union[None, bool, List[StereoComponent]] = None,
                  decode_fn: Optional[Callable] = None,
                  name: Optional[str] = None
                  ) -> List[NNComponent]:
        if len(inputs) != len(self.devices):
            raise RuntimeError(f'Number of inputs ({len(inputs)}) must match number of devices ({len(self.devices)})')

        nns = []
        if type(spatials) is not List:
            spatials = [spatials] * len(self.devices)

        for input, device, spatial in zip(inputs, self.devices, spatials):
            nns.append(device.create_nn(model, input, nn_type, tracker, spatial, decode_fn, name))

        return nns

    def stereo(self,
               resolution: Union[None, str, dai.MonoCameraProperties.SensorResolution] = None,
               fps: Optional[float] = None,
               lefts: Union[None, List[CameraComponent]] = None,  # Left mono camera
               rights: Union[None, List[CameraComponent]] = None,  # Right mono camera
               name: Optional[str] = None,
               encode: Union[None, str, bool, dai.VideoEncoderProperties.Profile] = None
               ) -> List[StereoComponent]:

        if lefts is None and rights is None:
            return [device.stereo(resolution, fps, lefts, rights, name, encode) for device in self.devices]

        stereo = []
        for device, left, right in zip(self.devices, lefts, rights):
            stereo.append(device.stereo(resolution, fps, left, right, name, encode))

        return stereo

    def create_imu(self) -> List[IMUComponent]:
        return [device.create_imu() for device in self.devices]

    def start(self, blocking=False):

        


        for device in self.devices:
            device.start(blocking=False)

        stop = False
        if blocking:
            while not stop:
                for device in self.devices:
                    stop = stop or not device.running()
                    device.poll()

    def poll(self) -> Optional[int]:
        for poll in self._polling:
            poll()

        key = None
        for device in self.devices:
            key = key or device.poll()
        return key

    def record(self,
            outputs: Union[Callable, List[Callable]],
            path: str,
            record_type: RecordType = RecordType.VIDEO) -> RecordPacketHandler:

        outputs = flatten(outputs)
        print('outputs',outputs)

        handler = RecordPacketHandler(outputs, Record(Path(path).resolve(), record_type))
        self._packet_handlers.append(handler)
        return handler

    def visualize(self,
                  outputs: Union[List, Callable, Component],
                  record_path: Optional[str] = None,
                  scale: float = None,
                  fps=False,
                  callback: Callable = None,
                  visualizer: str = 'opencv'
                  ) -> Visualizer:
        """
        Visualize component output(s). This handles output streaming (OAK->host), message syncing, and visualizing.
        Args:
            output (Component/Component output): Component output(s) to be visualized. If component is passed, SDK will visualize its default output (out())
            record_path: Path where to store the recording (visualization window name gets appended to that path), supported formats: mp4, avi
            scale: Scale the output window by this factor
            fps: Whether to show FPS on the output window
            callback: Instead of showing the frame, pass the Packet to the callback function, where it can be displayed
        """
        if type(outputs) is list:
            outputs = flatten(outputs)

        main_thread = False
        visualizer = visualizer.lower()
        if visualizer in ['opencv', 'cv2']:
            from depthai_sdk.visualize.visualizers.opencv_visualizer import OpenCvVisualizer
            vis = OpenCvVisualizer(scale, fps)
            main_thread=True # OpenCV's imshow() requires to be called from the main thread
        elif visualizer in ['depthai-viewer', 'depthai_viewer', 'viewer', 'depthai']:
            from depthai_sdk.visualize.visualizers.viewer_visualizer import DepthaiViewerVisualizer
            vis = DepthaiViewerVisualizer(scale, fps)
        elif visualizer in ['robothub', 'rh']:
            raise NotImplementedError('Robothub visualizer is not implemented yet')
        else:
            raise ValueError(f"Unknown visualizer: {visualizer}. Options: 'opencv'")

        handler = VisualizePacketHandler(outputs, vis, callback=callback, record_path=record_path, main_thread=main_thread)
        self._packet_handlers.append(handler)

        if main_thread:
            self._polling.append(handler._poll)

        return vis






