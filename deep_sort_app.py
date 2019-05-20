# vim: expandtab:ts=4:sw=4
from __future__ import division, print_function, absolute_import
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

import argparse
import os

import cv2
import numpy as np

from application_util import preprocessing
from application_util import visualization
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
import json
from tqdm import tqdm


def gather_sequence_info(image_dir, groundtruth, detections):
    """Gather sequence information, such as image filenames, detections,
    groundtruth (if available).

    Parameters
    ----------
    image_dir : str
        Path to the BDD tracking dataset directory.
    detection_dir : str
        Path to the detection file.

    Returns
    -------
    Dict
        A dictionary of the following sequence information:

        * sequence_name: Name of the sequence
        * image_filenames: A dictionary that maps frame indices to image
          filenames.
        * detections: A numpy array of detections in MOTChallenge format.
        * groundtruth: A numpy array of ground truth in MOTChallenge format.
        * image_size: Image size (height, width).
        * min_frame_idx: Index of the first frame.
        * max_frame_idx: Index of the last frame.

    """
    
    sequence_name = groundtruth[0]['video_name']
    image_size = (720, 1280, 3)
    frame_idxs = [g['index'] for g in groundtruth]
    image_filenames = [os.path.join(image_dir, g['name']) for g in groundtruth]
    min_frame_idx, max_frame_idx = min(frame_idxs), max(frame_idxs)
    update_ms = 1000 / 5

    feature_dim = detections.shape[1] - 10 if detections is not None else 0
    seq_info = {
        "sequence_name": sequence_name,
        "image_filenames": image_filenames,
        "detections": detections,
        "groundtruth": groundtruth,
        "image_size": image_size,
        "min_frame_idx": min_frame_idx,
        "max_frame_idx": max_frame_idx,
        "feature_dim": feature_dim,
        "update_ms": update_ms
    }
        
    return seq_info


def create_detections(detection_mat, frame_idx, min_height=0):
    """Create detections for given frame index from the raw detection matrix.

    Parameters
    ----------
    detection_mat : ndarray
        Matrix of detections. The first 10 columns of the detection matrix are
        in the standard MOTChallenge detection format. In the remaining columns
        store the feature vector associated with each detection.
    frame_idx : int
        The frame index.
    min_height : Optional[int]
        A minimum detection bounding box height. Detections that are smaller
        than this value are disregarded.

    Returns
    -------
    List[tracker.Detection]
        Returns detection responses at given frame index.

    """
    frame_indices = detection_mat[:, 0].astype(np.int)
    mask = frame_indices == frame_idx

    detection_list = []
    for row in detection_mat[mask]:
        bbox, confidence, feature = row[2:6], row[6], row[10:]
        if bbox[3] < min_height:
            continue
        detection_list.append(Detection(bbox, confidence, feature))
    return detection_list


def run(image_dir, groundtruth, detections, min_confidence,
        nms_max_overlap, min_detection_height, max_cosine_distance,
        nn_budget, display):
    """Run multi-target tracker on a particular sequence.

    Parameters
    ----------
    image_dir : str
        Path to the MOTChallenge sequence directory.
    detection_dir : str
        Path to the detections file.
    min_confidence : float
        Detection confidence threshold. Disregard all detections that have
        a confidence lower than this value.
    nms_max_overlap: float
        Maximum detection overlap (non-maxima suppression threshold).
    min_detection_height : int
        Detection height threshold. Disregard all detections that have
        a height lower than this value.
    max_cosine_distance : float
        Gating threshold for cosine distance metric (object appearance).
    nn_budget : Optional[int]
        Maximum size of the appearance descriptor gallery. If None, no budget
        is enforced.
    display : bool
        If True, show visualization of intermediate tracking results.

    """
    seq_info = gather_sequence_info(image_dir, groundtruth, detections)
    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    results = []

    def frame_callback(vis, frame_idx):
        print("Processing frame %05d" % frame_idx)

        # Load image and generate detections.
        detections = create_detections(
            seq_info["detections"], frame_idx, min_detection_height)
        detections = [d for d in detections if d.confidence >= min_confidence]

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(
            boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Update tracker.
        tracker.predict()
        tracker.update(detections)

        # Update visualization.
        if display:
            image = cv2.imread(
                seq_info["image_filenames"][frame_idx], cv2.IMREAD_COLOR)
            vis.set_image(image.copy())
            vis.draw_detections(detections)
            vis.draw_trackers(tracker.tracks)

        # Store results.
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlwh()
            results.append([
                frame_idx, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]])

    # Run tracker.
    if display:
        visualizer = visualization.Visualization(seq_info, update_ms=5)
    else:
        visualizer = visualization.NoVisualization(seq_info)
    visualizer.run(frame_callback)
    
    out = []
    curr_idx = -1
    img = None
    for row in results:
        if curr_idx != row[0]:
            if img != None:
                out += [img]
            img = {
                    'videoName': video_name,
                    'name': '{}-{}.jpg'.format(video_name, str(row[0]).zfill(6)),
                    'index': row[0],
                    'labels': []
                }
            curr_idx = row[0]
            
        img['labels'] += [{
                            'category': None,
                            'box2d': {
                                        'x1': row[2], 'y1': row[3],
                                        'x2': row[2] + row[4],
                                        'y2': row[3] + row[5]
                                     },
                            'id': row[1]
                        }]
    
    out += [img]
    
    return out
        
                        
def bool_string(input_string):
    if input_string not in {"True","False"}:
        raise ValueError("Please Enter a valid Ture/False choice")
    else:
        return (input_string == "True")


def get_gt_labels(gt_dir):
    
    with open(gt_dir) as f:
        gt_annos = json.load(f)
    
    # group by videos
    annos_by_video = {}
    for gt_anno in gt_annos:
        if gt_anno['video_name'] in annos_by_video.keys():
            annos_by_video[gt_anno['video_name']] += [gt_anno]
        else:
            annos_by_video[gt_anno['video_name']] = [gt_anno]
    
    return annos_by_video
    

def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Deep SORT")
    parser.add_argument(
        "--image_dir", help="Path to BDD tracking sequence directory",
        default=None, required=True)
    parser.add_argument(
        "--gt_dir", help="Path to the ground truth labels of the BDD tracking sequences",
        default=None, required=True)
    parser.add_argument(
        "--detection_dir", help="Path to custom detections.", default=None,
        required=True)
    parser.add_argument(
        "--output_dir", help="Path to the tracking output directory. This file will"
        " contain the tracking results on completion.",
        default="/tmp/hypotheses")
    parser.add_argument(
        "--min_confidence", help="Detection confidence threshold. Disregard "
        "all detections that have a confidence lower than this value.",
        default=0.8, type=float)
    parser.add_argument(
        "--min_detection_height", help="Threshold on the detection bounding "
        "box height. Detections with height smaller than this value are "
        "disregarded", default=0, type=int)
    parser.add_argument(
        "--nms_max_overlap",  help="Non-maxima suppression threshold: Maximum "
        "detection overlap.", default=1.0, type=float)
    parser.add_argument(
        "--max_cosine_distance", help="Gating threshold for cosine distance "
        "metric (object appearance).", type=float, default=0.2)
    parser.add_argument(
        "--nn_budget", help="Maximum size of the appearance descriptors "
        "gallery. If None, no budget is enforced.", type=int, default=None)
    parser.add_argument(
        "--display", help="Show intermediate tracking results",
        default=True, type=bool_string)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    annos_by_video = get_gt_labels(args.gt_dir)
    
    out = []
    for detection_file in tqdm(os.listdir(args.detection_dir)):
        detection_dir = os.path.join(args.detection_dir, detection_file)
        video_name = detection_file.split('.')[0]
        groundtruth = annos_by_video[video_name]
        detections = np.load(detection_dir)
        
        out += [run(
            args.image_dir, groundtruth, detections,
            args.min_confidence, args.nms_max_overlap, args.min_detection_height,
            args.max_cosine_distance, args.nn_budget, args.display)]
            
    with open(args.output_dir, 'w') as f:
        json.dump(out, f)
