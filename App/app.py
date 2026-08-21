from flask import Flask, jsonify, render_template, send_file
from pathlib import Path
from data.reader import get_sequence, get_dataset
from evaluation.eval import get_channel, get_forecast_channel
from evaluation.stats import calculate_statistics

app = Flask(__name__)
app.config["SECRET_KEY"] = "dev"

get_dataset("MOT20")


@app.route("/api/models")
def get_models():
    """
       Main writer:Néo Deward
       Reviewer:
       Contributors:

       Serve the main GUI page find the pt file for every transformer or ltsm.
       """
    project_root = Path(__file__).resolve().parents[1]
    checkpoints_dir = project_root / "checkpoints"


    models = [{"id": "kalman", "name": "Kalman Filter (Baseline)"}]

    if checkpoints_dir.exists():
        for pt_file in checkpoints_dir.rglob("*.pt"):
            models.append({
                "id": pt_file.stem,
                "name": f"{pt_file.stem}"
            })

    return jsonify({"models": models})
@app.route("/")
def index():
    """
    Main writer: Keez Cuijpers
    Reviewer: Noah Nuelandt
    Contributors: Keez Cuijpers

    Serve the main GUI page, including the model switcher for Kalman,
    Transformer, and real versus synthetic benchmark models.
    """
    return render_template("index.html")

@app.route("/api/sequences/<sequence>/channels/<channel>/tracks")
def get_ground_truth(sequence: str, channel: str):
    """
    Main writer: Keez Cuijpers
    Reviewer: Noah Nuelandt
    Contributors: Claire Bams

    Return ground-truth tracks for a sequence and channel.
    """
    seq = get_sequence(sequence)

    channel_data = get_channel(seq, channel)

    tracks = []
    for track in channel_data.tracks.values():
        track_dict = {
            "id": int(track.id),
            "points": []
        }
        for frame_id, ann in track.gt.items():
            track_dict["points"].append({
                "frame_id": int(frame_id),
                "x": ann.bbox.foot_x,
                "y": ann.bbox.foot_y
            })
        tracks.append(track_dict)

    return jsonify({"tracks": tracks})


@app.route("/api/sequences/<sequence>/channels/<channel>/forecasts")
def get_forecasts(sequence: str, channel: str):
    """
    Main writer: Keez Cuijpers
    Reviewer: Noah Nuelandt
    Contributors: Claire Bams

    Return forecast tracks for a sequence and channel.
    """
    seq = get_sequence(sequence)
    forecast_channel = get_forecast_channel(seq, channel)

    tracks = []
    for track in forecast_channel.tracks.values():
        tracks.append(
            {
                "id": int(track.id),
                "forecasts": [
                    {
                        "anchor_frame_id": int(forecast.anchor_frame_id),
                        "x": forecast.x,
                        "y": forecast.y,
                        "points": [
                            {
                                "frame_id": int(point.frame_id),
                                "horizon": int(point.horizon),
                                "x": point.x,
                                "y": point.y,
                            }
                            for point in forecast.points
                        ],
                    }
                    for forecast in track.forecasts
                ],
            },
        )

    return jsonify({"tracks": tracks})


@app.route("/api/sequences/<sequence>/channels/<channel>/statistics")
def get_statistics(sequence: str, channel: str):
    """
    Main writer: Keez Cuijpers
    Reviewer: Noah Nuelandt
    Contributors: Claire Bams

    Return summary statistics for ground truth versus the selected channel.
    """
    seq = get_sequence(sequence)

    # Ensure both GT and requested channel are loaded

    gt_channel = get_channel(seq, "gt")
    pred_channel = get_channel(seq, channel)

    stats = calculate_statistics(gt_channel, pred_channel)
    return jsonify(stats)


@app.route("/api/sequences/<sequence>/manifest")
def get_frame_manifest(sequence: str):
    """
    Main writer: Keez Cuijpers
    Reviewer: Noah Nuelandt
    Contributors: Claire Bams

    Return sequence metadata for the frame viewer.
    """
    seq = get_sequence(sequence)
    info = seq.info["Sequence"]

    return jsonify({
        "fps": int(info.get("frameRate", 25)),
        "width": int(info.get("imWidth", 1920)),
        "height": int(info.get("imHeight", 1080)),
        "frame_count": len(seq.frames)
    })


@app.route("/api/sequences/<sequence>/frames/<int:frame_id>/img")
def get_frame_image(sequence: str, frame_id: int):
    """
    Main writer: Keez Cuijpers
    Reviewer: Noah Nuelandt
    Contributors: Claire Bams

    Return the image for a specific frame.
    """
    seq = get_sequence(sequence)
    image_path = seq.frames[frame_id].path
    if not image_path.exists() or not image_path.is_file():
        return {"error": f"Image file not found for frame {frame_id}."}, 404

    return send_file(image_path)

if __name__ == "__main__":
    app.run(debug=True)