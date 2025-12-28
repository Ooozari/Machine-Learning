import cv2
import numpy as np
from models.yolo_model import yolo_model


def run_object_detection(image_path):
    results = yolo_model(image_path)
    return parse_boxes(results)


def run_classification(image_path):
    results = yolo_model(image_path, task="classify")
    return results[0].probs.data.tolist()


def run_segmentation(image_path):
    results = yolo_model(image_path, task="segment")
    return parse_masks(results)


def run_instance_segmentation(image_path):
    results = yolo_model(image_path, task="segment")
    return parse_boxes(results, with_masks=True)


def run_pose_estimation(image_path):
    results = yolo_model(image_path, task="pose")
    return parse_keypoints(results)


def parse_boxes(results, with_masks=False):
    output = []
    r = results[0]
    names = r.names

    for i, box in enumerate(r.boxes):
        item = {
            "class_id": int(box.cls),
            "class_name": names[int(box.cls)],
            "confidence": float(box.conf),
            "bbox": box.xyxy[0].tolist()
        }

        if with_masks and r.masks:
            item["mask"] = r.masks.xy[i].tolist()

        output.append(item)

    return output



def parse_masks(results):
    masks = results[0].masks
    if masks is None:
        return []

    return [mask.tolist() for mask in masks.xy]


def parse_keypoints(results):
    kps = results[0].keypoints
    if kps is None:
        return []

    return kps.xy.tolist()
