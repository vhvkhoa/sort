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
        default='AIC20_track1/Dataset_A/'
    )
    parser.add_argurment(
        'video-extension',
        type=str,
        default='mp4'
    )
    parser.add_argument(
        '--input-bbox-dir',
        type=str,
        default='bboxes/'
    )
    parser.add_argument(
        '--output-video-dir',
        type=str,
        default='output_videos/'
    )
    parser.add_argument(
        '--iou-thresh',
        type=float,
        default=0.3
    )
    parser.add_argument(
        '--confidence-thresh',
        type=float,
        default=0.3
    )
    parser.add_argument(
        '--time-thresh',
        type=int,
        default=100
    )
    return parser.parse_args()


def draw_bbox(frame, bbox, bbox_id):
    bbox = np.array(bbox, dtype=np.int32)
    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
    cv2.putText(
        frame, str(bbox_id),
        (bbox[0], bbox[1] - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5, (0, 255, 0), thickness=1, lineType=cv2.LINE_AA)
    return frame


def main(args):
    input_video_paths = glob(path.join(args.input_video_dir, '*.' + args.video_extension))

    for input_video_path in input_video_paths:
        with open(path.join(args.input_bbox_dir, input_video_path + '.pkl'), 'rb') as f:
            bboxes = pkl.load(f)

        input_video = cv2.VideoCapture(input_video_path)
        width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        output_video = cv2.VideoWriter(
            filename=path.join(args.output_video_dir, input_video_path),
            fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
            fps=30.,
            frameSize=(width, height),
            isColor=True
        )
        num_frames = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))

        trackers = []
        tracked_bboxes = []
        bbox_ids = []
        current_bbox_id = 0

        for frame_idx in tqdm(range(num_frames)):
            success, frame = input_video.read()

            # Keep cars and trucks
            frame_bboxes = np.concatenate([
                bboxes[frame_idx][3],
                bboxes[frame_idx][8]],
                axis=0
            )

            # Keep bboxes with confidence score more than threshold
            frame_bboxes = [
                bbox[:4].astype(np.int32) for bbox in frame_bboxes
                if bbox[4] > args.confidence_thresh
            ]

            # Remove bboxes that cannot be tracked or exists over a threshold
            untracked_ids = []
            for i, tracker in enumerate(trackers):
                success, bbox = tracker.update(frame)
                if success:  # and frame_idx - start_frame < args.time_thresh:
                    tracked_bboxes[i] = np.array([bbox[0], bbox[1], bbox[2] + bbox[0], bbox[3] + bbox[1]])
                else:
                    untracked_ids.append(i)
            if len(untracked_ids) > 0:
                for index in untracked_ids[::-1]:
                    del tracked_bboxes[index]
                    del trackers[index]
                    del bbox_ids[index]

            if len(frame_bboxes) > 0 and len(tracked_bboxes) > 0:
                ious = mask_util.iou(np.array(frame_bboxes), np.array(tracked_bboxes), np.zeros((len(tracked_bboxes),), dtype=np.bool))
            elif len(frame_bboxes) > 0:
                ious = np.zeros((len(frame_bboxes), 1))

            max_iou_per_new = np.asarray(ious).max(axis=1).tolist()
            for iou, bbox in zip(max_iou_per_new, frame_bboxes):
                if iou <= args.iou_thresh:
                    tracked_bboxes.append(bbox)
                    bbox_ids.append(current_bbox_id)
                    trackers.append(cv2.TrackerMOSSE_create())
                    bbox = (bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1])
                    trackers[-1].init(frame, bbox)
                    current_bbox_id += 1

            for bbox, bbox_id in zip(tracked_bboxes, bbox_ids):
                frame = draw_bbox(frame, bbox, bbox_id)
            output_video.write(frame)


if __name__ == '__main__':
    args = get_args()
    main(args)
