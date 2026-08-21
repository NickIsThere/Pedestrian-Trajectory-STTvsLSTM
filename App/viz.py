import cv2

from data.reader import get_dataset, Frame, Sequence
from typing import Callable
import numpy as np
from tqdm import tqdm
from evaluation.eval import _ensure_transformer_loaded, get_denormalize
from utils.modelutils import prepare_scene_inference_batch
import torch

DISPLAY_HORIZON_FRAMES = 50

def generate_visualizations(callback: Callable[[np.ndarray, Frame, Sequence]]) -> None:
    """
    Main writer: Noah Nuelandt
    Reviewer: 
    Contributors:
    """
    dataset = get_dataset("MOT20")
    for split in dataset.splits.values():
        for sequence in split.sequences.values():
            video_name = f"{sequence.name}.mp4"
            seq_info = sequence.info["Sequence"]
            fps = int(seq_info.get("frameRate", 25))
            width = int(seq_info.get("imWidth", 1920))
            height = int(seq_info.get("imHeight", 1080))
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

            for frame in tqdm(sorted(sequence.frames.values(), key=lambda f: f.id), desc=f"{sequence.name}", unit="frame", leave=False):
                image = cv2.imread(str(frame.path))
                if image is None:
                    continue
                image = callback(image, frame, sequence)
                video_writer.write(image)
            video_writer.release()

            break
        break

model = None
cfg = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_transformer_loaded = False


def _get_transformer():
    global model, cfg, device, _transformer_loaded
    if not _transformer_loaded:
        model, cfg, device = _ensure_transformer_loaded()
        _transformer_loaded = True
    return model, cfg, device


def draw_points(image: np.ndarray, frame: Frame, sequence: Sequence) -> np.ndarray:
    """
    Main writer: Noah Nuelandt
    Reviewer: 
    Contributors: Nick Grebe
    """
    denormalize = get_denormalize(sequence)
    transformer_model, transformer_cfg, transformer_device = _get_transformer()
    for annotation in frame.det.values():
        x = int(annotation.bbox.x)
        y = int(annotation.bbox.y)
        x2 = int(annotation.bbox.x + annotation.bbox.width)
        y2 = int(annotation.bbox.y + annotation.bbox.height)
        #image = cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
        image = cv2.rectangle(image, (x, y), (x2, y2), (0, 0, 255), 1)
        image = cv2.putText(image, str(annotation.track_id), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        #predicted_annotation = sequence.channels["transformer"].tracks[annotation.track_id].gt[frame.id]
        #x = int(predicted_annotation.bbox.foot_x)
        #y = int(predicted_annotation.bbox.foot_y)
        #image = cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

    if transformer_model is None or transformer_cfg is None:
        return image

    prepared, track_ids = prepare_scene_inference_batch(
        sequence,
        frame.id,
        device=transformer_device,
        config=transformer_cfg,
        source="det",
    )
    if not track_ids:
        return image

    with torch.no_grad():
        pred = transformer_model(**prepared)

    display_steps = min(
        transformer_cfg.future_steps,
        max(1, DISPLAY_HORIZON_FRAMES // transformer_cfg.trajectory_stride),
    )

    for agent_index, track_id in enumerate(track_ids):
        ann = frame.det[track_id]
        anchor = (int(ann.bbox.foot_x), int(ann.bbox.foot_y))
        path = [anchor]
        for i in range(display_steps):
            px_n = float(pred["pred_positions"][0, agent_index, i, 0])
            py_n = float(pred["pred_positions"][0, agent_index, i, 1])
            px, py = denormalize(px_n, py_n)
            path.append((int(px), int(py)))

        if len(path) < 2:
            continue

        cv2.polylines(image, [np.array(path, dtype=np.int32)], False, (0, 255, 0), 2)
        for point in path[1:]:
            image = cv2.circle(image, point, 3, (0, 255, 0), -1)
    return image

if __name__ == "__main__":
    generate_visualizations(draw_points)
