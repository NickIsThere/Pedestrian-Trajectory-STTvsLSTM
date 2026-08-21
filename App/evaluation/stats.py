from math import hypot
from data.reader import Channel

def calculate_statistics(ground_truth: Channel, prediction: Channel) -> dict:
    """
    initial work by: nick grebe - i6377605
    Refactor by noah nuelandt - i6375705
    """
    observation_errors: list[float] = []
    final_track_errors: list[float] = []
    person_count = 0

    for track_id, gt_track in ground_truth.tracks.items():
        if track_id not in prediction.tracks:
            continue

        person_count += 1
        pred_track = prediction.tracks[track_id]

        track_errors: list[float] = []
        last_error: float | None = None

        frames = sorted(set(gt_track.gt.keys()).intersection(pred_track.gt.keys()))

        if len(frames) < 2:
            continue

        for frame_id in frames[1:]:
            gt_ann = gt_track.gt[frame_id]
            pred_ann = pred_track.gt[frame_id]

            gx = gt_ann.bbox.foot_x
            gy = gt_ann.bbox.foot_y

            px = pred_ann.bbox.foot_x
            py = pred_ann.bbox.foot_y

            error = hypot(gx - px, gy - py)
            track_errors.append(error)
            last_error = error

        if track_errors:
            observation_errors.extend(track_errors)
        if last_error is not None:
            final_track_errors.append(last_error)

    ade = float(sum(observation_errors) / len(observation_errors)) if observation_errors else 0.0
    fde = float(sum(final_track_errors) / len(final_track_errors)) if final_track_errors else 0.0

    return {
        "person_count": person_count,
        "ade": ade,
        "fde": fde,
    }
