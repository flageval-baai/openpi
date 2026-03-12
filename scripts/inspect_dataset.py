"""Script to inspect LeRobot dataset: save actions to JSON and combine camera images to video."""

import json
import argparse
import subprocess
import tempfile
import shutil
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def save_actions_to_json(dataset: LeRobotDataset, output_path: Path):
    """Save all actions from the dataset to a JSON file."""
    actions_data = {
        "metadata": {
            "num_frames": len(dataset),
            "num_episodes": dataset.num_episodes,
            "fps": dataset.fps,
        },
        "frames": []
    }

    # Find the action key
    action_key = None
    sample = dataset[0]
    for key in ["action", "actions"]:
        if key in sample:
            action_key = key
            break

    if action_key is None:
        raise ValueError(f"No action key found in dataset. Available keys: {list(sample.keys())}")

    print(f"Using action key: {action_key}")
    print(f"Action shape: {sample[action_key].shape}")

    for idx in tqdm(range(len(dataset)), desc="Extracting actions"):
        sample = dataset[idx]
        frame_data = {
            "index": int(sample["index"]),
            "episode_index": int(sample["episode_index"]),
            "frame_index": int(sample["frame_index"]),
            "timestamp": float(sample["timestamp"]),
            action_key: sample[action_key].tolist(),
        }

        # Also save state if available
        for state_key in ["observation.state", "state"]:
            if state_key in sample:
                frame_data[state_key] = sample[state_key].tolist()
                break

        actions_data["frames"].append(frame_data)

    with open(output_path, "w") as f:
        json.dump(actions_data, f, indent=2)

    print(f"Actions saved to {output_path}")


def create_combined_video(dataset: LeRobotDataset, output_path: Path):
    """Create a video combining all camera views side by side using ffmpeg."""
    # Get camera keys
    camera_keys = dataset.meta.camera_keys
    print(f"Found camera keys: {camera_keys}")

    if len(camera_keys) == 0:
        print("No camera keys found in dataset!")
        return

    # Get sample to determine image dimensions
    sample = dataset[0]

    # Get image shapes for each camera
    img_shapes = {}
    for key in camera_keys:
        if key in sample:
            img = sample[key]
            if isinstance(img, torch.Tensor):
                img_shapes[key] = img.shape
            print(f"  {key}: shape={img.shape if isinstance(img, torch.Tensor) else type(img)}")

    if len(img_shapes) == 0:
        print("No valid image data found!")
        return

    # Use up to 3 cameras (or all if fewer)
    selected_cameras = list(img_shapes.keys())[:3]
    print(f"Using cameras: {selected_cameras}")

    # Get dimensions (assuming CHW format)
    first_shape = img_shapes[selected_cameras[0]]
    if len(first_shape) == 3:
        # CHW format
        _, h, w = first_shape
    else:
        print(f"Unexpected image shape: {first_shape}")
        return

    # Combined frame dimensions (horizontal concatenation)
    combined_w = w * len(selected_cameras)
    combined_h = h
    fps = int(dataset.fps)

    print(f"Creating video: {combined_w}x{combined_h} @ {fps}fps")

    # Create temporary directory for frames
    tmp_dir = tempfile.mkdtemp()
    try:
        print(f"Saving frames to temporary directory...")
        for idx in tqdm(range(len(dataset)), desc="Extracting frames"):
            sample = dataset[idx]

            frames = []
            for key in selected_cameras:
                img = sample[key]

                if isinstance(img, torch.Tensor):
                    # Convert from CHW to HWC and from tensor to numpy
                    img = img.numpy()
                    if img.shape[0] == 3:  # CHW format
                        img = np.transpose(img, (1, 2, 0))

                    # Handle different value ranges
                    if img.max() <= 1.0:
                        img = (img * 255).astype(np.uint8)
                    else:
                        img = img.astype(np.uint8)

                frames.append(img)

            # Combine frames horizontally
            combined_frame = np.concatenate(frames, axis=1)

            # Save frame as PNG
            frame_path = Path(tmp_dir) / f"frame_{idx:06d}.png"
            Image.fromarray(combined_frame).save(frame_path)

        # Use ffmpeg to create video
        print("Encoding video with ffmpeg...")
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", f"{tmp_dir}/frame_%06d.png",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "23",
            str(output_path)
        ]

        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"ffmpeg error: {result.stderr}")
            raise RuntimeError("ffmpeg encoding failed")

        print(f"Video saved to {output_path}")

    finally:
        # Clean up temporary directory
        shutil.rmtree(tmp_dir)


def main():
    parser = argparse.ArgumentParser(description="Inspect LeRobot dataset")
    parser.add_argument("--root", type=str, required=True, help="Path to dataset root")
    parser.add_argument("--repo-id", type=str, required=True, help="Dataset repo ID")
    parser.add_argument("--episodes", type=int, nargs="+", default=[0], help="Episode indices to load")
    parser.add_argument("--output-dir", type=str, default="./dataset_inspection", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from {args.root}")
    print(f"Repo ID: {args.repo_id}")
    print(f"Episodes: {args.episodes}")

    dataset = LeRobotDataset(
        repo_id=args.repo_id,
        root=args.root,
        episodes=args.episodes,
    )

    print(f"\nDataset info:")
    print(f"  Total frames: {len(dataset)}")
    print(f"  Episodes: {dataset.num_episodes}")
    print(f"  FPS: {dataset.fps}")
    print(f"  Camera keys: {dataset.meta.camera_keys}")

    # Save actions to JSON
    actions_path = output_dir / "actions.json"
    save_actions_to_json(dataset, actions_path)

    # Create combined video
    video_path = output_dir / "combined_video.mp4"
    create_combined_video(dataset, video_path)

    print(f"\nDone! Output saved to {output_dir}")


if __name__ == "__main__":
    main()
