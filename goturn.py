import argparse
import pickle as pkl

from tqdm import tqdm
import cv2
import numpy as np
import pycocotools.mask as mask_util


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input-video',
        type=str,
        default='../datasets/AIC20_track1/Dataset_A/cam_1.mp4'
    )
    parser.add_argument(
        '--input-bbox',
        type=str,
        default='bboxes/cam_1.mp4.pkl'
    )
    parser.add_argument(
        '--output-video',
        type=str,
        default='output_videos/cam_1.mp4'
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
    return parser.parse_args()


def draw_bbox(frame, bbox):
    bbox = np.array(bbox, dtype=np.int32)
    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
    return frame


def bbox_exists(bbox, tracked_bboxes):
    pass


def main(args):
    with open(args.input_bbox, 'rb') as f:
        bboxes = pkl.load(f)

    input_video = cv2.VideoCapture(args.input_video)
    width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_video = cv2.VideoWriter(
        filename=args.output_video,
        fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
        fps=30.,
        frameSize=(width, height),
        isColor=True
    )
    num_frames = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))

    trackers = []
    tracked_bboxes = []
    for frame_idx in tqdm(range(num_frames)):
        success, frame = input_video.read()
        frame_bboxes = np.concatenate([
            bboxes[frame_idx][3],
            bboxes[frame_idx][8]],
            axis=0
        ).tolist()
        frame_bboxes = [
            tuple(bbox[:4]) for bbox in frame_bboxes
            if bbox[4] > args.confidence_thresh
        ]

        untracked_ids = []
        for i, tracker in enumerate(trackers):
            success, bbox = tracker.update(frame)
            if success:
                tracked_bboxes[i] = bbox
            else:
                untracked_ids.append(i)
        if len(untracked_ids) > 0:
            for index in untracked_ids[::-1]:
                del tracked_bboxes[index]

        if len(frame_bboxes) > 0 and len(tracked_bboxes) > 0:
            ious = mask_util.iou(np.array(frame_bboxes), np.array(tracked_bboxes), np.zeros((len(tracked_bboxes),), dtype=np.bool))
        elif len(frame_bboxes) > 0:
            ious = np.zeros((len(frame_bboxes), 1))

        max_iou_per_new = np.asarray(ious).max(axis=1).tolist()
        for iou, bbox in zip(max_iou_per_new, frame_bboxes):
            if iou <= args.iou_thresh:
                trackers.append(cv2.TrackerGOTURN_create())
                trackers[-1].init(frame, bbox)
                tracked_bboxes.append(bbox)

        for bbox in tracked_bboxes:
            frame = draw_bbox(frame, bbox)
        output_video.write(frame)


if __name__ == '__main__':
    args = get_args()
    main(args)
