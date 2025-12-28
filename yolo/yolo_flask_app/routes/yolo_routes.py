import os
from flask import Blueprint, request, jsonify
from config import UPLOAD_FOLDER
from services.yolo_service import (
    run_object_detection,
    run_classification,
    run_segmentation,
    run_instance_segmentation,
    run_pose_estimation
)

yolo_routes = Blueprint("yolo_routes", __name__)


def save_file(file):
    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)
    return path


@yolo_routes.route("/detect", methods=["POST"])
def detect():
    file = request.files["file"]
    path = save_file(file)
    return jsonify(run_object_detection(path))


@yolo_routes.route("/classify", methods=["POST"])
def classify():
    file = request.files["file"]
    path = save_file(file)
    return jsonify(run_classification(path))


@yolo_routes.route("/segment", methods=["POST"])
def segment():
    file = request.files["file"]
    path = save_file(file)
    return jsonify(run_segmentation(path))


@yolo_routes.route("/instance-segment", methods=["POST"])
def instance_segment():
    file = request.files["file"]
    path = save_file(file)
    return jsonify(run_instance_segmentation(path))


@yolo_routes.route("/pose", methods=["POST"])
def pose():
    file = request.files["file"]
    path = save_file(file)
    return jsonify(run_pose_estimation(path))
