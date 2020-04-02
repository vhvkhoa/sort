from os import path
from glob import glob
import argparse
import pickle as pkl

from tqdm import tqdm
import cv2
import numpy as np
import pycocotools.mask as mask_util


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input-video-dir',
        type=str,
        default='../datasets/AIC20_track1/Dataset_A/'
    )
    parser.add_argument(
        '--video-extension',
        type=str,
        default='mp4'
    )
    parser.add_argument(
        '--input-bbox-dir',
        type=str,
        default='bboxes/'
    )
    parser.add_argument(
        '--input-roi-dir',
        type=str,
        default='ROI_numpy/'
    )
    parser.add_argument(
        '--output-video-dir',
        type=str,
        default='MOSSE_track_output/'
    )
    parser.add_argument(
        '--iou-thresh',
        type=float,
        default=0.6
    )
    parser.add_argument(
        '--confidence-thresh',
        type=float,
        default=0.3
    )
    parser.add_argument(
        '--time-thresh',
        type=int,
        default=30
    )
    parser.add_argument(
        '--dist-thresh',
        type=int,
        default=10
    )
    return parser.parse_args()


def draw_bbox(frame, bbox, bbox_id, roi):
    bbox = np.array(bbox, dtype=np.int32)
    roi = np.array(roi, np.int32).reshape(-1, 1, 2)

    cv2.polylines(frame, [roi], True, (0, 255, 0))
    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
    cv2.putText(
        frame, str(bbox_id),
        (bbox[0], bbox[1] - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5, (0, 255, 0), thickness=1, lineType=cv2.LINE_AA)
    return frame


def verify_bbox(roi, bbox, old_bboxes=None, dist_thresh=None, time_thresh=None):
    center_x = float(bbox[2] - bbox[0]) / 2
    center_y = float(bbox[3] - bbox[1]) / 2

    result = roi[int(center_x)][int(center_y)]

    if old_bboxes is None or len(old_bboxes) < time_thresh or result is False:
        return result

    old_bbox = old_bboxes[-time_thresh]
    old_center_x = float(old_bbox[2] - bbox[0]) / 2
    old_center_y = float(old_bbox[3] - old_bbox[1]) / 2

    l1_dist = abs(center_x - old_center_x) + abs(center_y - old_center_y)
    if l1_dist > dist_thresh:
        result = True

    return result


def main(args):
    input_video_paths = glob(path.join(args.input_video_dir, '*.' + args.video_extension))

    for input_video_path in input_video_paths:
        print(input_video_path)
        cam_name = '_'.join(path.basename(input_video_path).split('_')[:2])
        with open(path.join(args.input_bbox_dir, path.basename(input_video_path) + '.pkl'), 'rb') as f:
            bboxes = pkl.load(f)
        with open(path.join(args.input_roi_dir, cam_name + '.txt')) as f:
            roi_coords = [[int(coord) for coord in line.split(',')] for line in f.read().split('\n')[:-1]]

        roi = np.load(path.join(args.input_roi_dir, cam_name + '.npy'))

        input_video = cv2.VideoCapture(input_video_path)
        width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        output_video = cv2.VideoWriter(
            filename=path.join(args.output_video_dir, path.basename(input_video_path)),
            fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
            fps=30.,
            frameSize=(width, height),
            isColor=True
        )
        num_frames = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))

        trackers = []
        tracked_bboxes = []
        start_times = []
        bbox_ids = []
        current_bbox_id = 0

        for frame_idx in tqdm(range(num_frames)):
            success, frame = input_video.read()

            # Keep cars and trucks
            frame_bboxes = np.concatenate([
                bboxes[frame_idx][1],
                bboxes[frame_idx][2]],
                axis=0
            )

            # Keep bboxes with confidence score more than threshold
            frame_bboxes = [
                bbox[:4].astype(np.int32) for bbox in frame_bboxes
                if bbox[4] > args.confidence_thresh
            ]

            # Remove bboxes that cannot be tracked or exists over a threshold
            untracked_ids = []
            for i, (tracker, start_idx) in enumerate(zip(trackers, start_times)):
                success, bbox = tracker.update(frame)
                bbox = [bbox[0], bbox[1], bbox[2] + bbox[0], bbox[3] + bbox[1]]
                if success and verify_bbox(roi, bbox, tracked_bboxes[i], args.dist_thresh, args.time_thresh):
                    tracked_bboxes[i].append(np.array([bbox[0], bbox[1], bbox[2] + bbox[0], bbox[3] + bbox[1]]))
                else:
                    untracked_ids.append(i)
            if len(untracked_ids) > 0:
                for index in untracked_ids[::-1]:
                    del tracked_bboxes[index]
                    del trackers[index]
                    del start_times[index]
                    del bbox_ids[index]

            if len(frame_bboxes) > 0 and len(tracked_bboxes) > 0:
                latest_bboxes = [tracked_car[-1] for tracked_car in tracked_bboxes]
                ious = mask_util.iou(np.array(frame_bboxes), np.array(latest_bboxes), np.zeros((len(latest_bboxes),), dtype=np.bool))
            elif len(frame_bboxes) > 0:
                ious = np.zeros((len(frame_bboxes), 1))

            max_iou_per_new = np.asarray(ious).max(axis=1).tolist()
            arg_max_iou_per_new = np.asarray(ious).argmax(axis=1).tolist()
            for iou, arg, bbox in zip(max_iou_per_new, arg_max_iou_per_new, frame_bboxes):
                if iou <= args.iou_thresh:
                    if verify_bbox(roi, bbox):
                        tracked_bboxes.append([bbox])
                        start_times.append(frame_idx)
                        bbox_ids.append(current_bbox_id)
                        trackers.append(cv2.TrackerMOSSE_create())
                        bbox = (bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1])
                        trackers[-1].init(frame, bbox)
                        current_bbox_id += 1
                else:
                    tracked_bboxes[arg][-1] = bbox
                    trackers[arg] = cv2.TrackerMOSSE_create()
                    bbox = (bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1])
                    trackers[arg].init(frame, bbox)

            for tracked_seq, bbox_id in zip(tracked_bboxes, bbox_ids):
                frame = draw_bbox(frame, tracked_seq[-1], bbox_id, roi_coords)
            output_video.write(frame)


if __name__ == '__main__':
    args = get_args()
    main(args)
