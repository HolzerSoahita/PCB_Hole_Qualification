import torch


class YOLOv5Singleton:
    _instance = None

    @staticmethod
    def get_instance(path):
        if not YOLOv5Singleton._instance:
            YOLOv5Singleton._instance = torch.hub.load(
                'ultralytics/yolov5', 'custom', path)
        return YOLOv5Singleton._instance
