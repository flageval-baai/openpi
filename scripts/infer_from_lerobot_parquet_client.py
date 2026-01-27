#!/usr/bin/env python
"""Infer one action chunk from a LeRobot parquet frame via openpi-client.

Reads a single episode parquet, finds the first gripper-close event, uses the
previous frame to build an observation, then queries a running policy server.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np

try:
    import cv2  # type: ignore
except Exception as exc:  # pragma: no cover - runtime dependency
    raise SystemExit("OpenCV (cv2) is required for video decoding.") from exc

try:
    from openpi_client import image_tools
    from openpi_client import msgpack_numpy
    from openpi_client import websocket_client_policy
except Exception as exc:  # pragma: no cover - runtime dependency
    raise SystemExit("openpi-client is required. Install packages/openpi-client.") from exc


def _find_dataset_root(parquet_path: Path) -> Path:
    cur = parquet_path.resolve()
    for parent in [cur.parent, *cur.parents]:
        meta = parent / "meta" / "info.json"
        if meta.exists():
            return parent
    raise FileNotFoundError("Could not locate dataset root with meta/info.json.")


def _load_info(dataset_root: Path) -> dict:
    with open(dataset_root / "meta" / "info.json", "r", encoding="utf-8") as f:
        return json.load(f)


def _parse_episode_index(parquet_path: Path) -> int:
    match = re.search(r"episode_(\d+)\.parquet$", parquet_path.name)
    if not match:
        raise ValueError(f"Could not parse episode index from {parquet_path}")
    return int(match.group(1))


def _load_parquet_columns(parquet_path: Path, columns: list[str]) -> dict[str, np.ndarray | list]:
    try:
        import pandas as pd  # type: ignore

        df = pd.read_parquet(parquet_path, columns=columns)
        out: dict[str, np.ndarray | list] = {}
        for col in columns:
            series = df[col]
            if series.dtype == object:
                out[col] = series.to_list()
            else:
                out[col] = series.to_numpy()
        return out
    except Exception:
        pass

    try:
        import pyarrow.parquet as pq  # type: ignore

        table = pq.read_table(parquet_path, columns=columns)
        out = {}
        for col in columns:
            arr = table.column(col)
            # Convert list-like columns to python lists for stacking.
            out[col] = arr.to_pylist()
        return out
    except Exception as exc:
        raise SystemExit("Install pandas or pyarrow to read parquet files.") from exc


def _stack_if_list(value):
    if isinstance(value, list):
        return np.stack([np.asarray(v) for v in value], axis=0)
    return np.asarray(value)


def _read_video_frame(video_path: Path, frame_index: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Failed to open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"Failed to read frame {frame_index} from {video_path}")
    # Convert BGR -> RGB
    return frame[:, :, ::-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet-path", required=True, type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--gripper-index", default=7, type=int)
    parser.add_argument("--gripper-threshold", default=0.7, type=float)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--image-front-key", default="observation.images.image_front")
    parser.add_argument("--image-wrist-key", default="observation.images.image_wrist")
    parser.add_argument("--disable-ping", action="store_true")
    args = parser.parse_args()

    parquet_path = args.parquet_path.resolve()
    dataset_root = _find_dataset_root(parquet_path)
    info = _load_info(dataset_root)
    episode_index = _parse_episode_index(parquet_path)

    columns = ["action", "observation.state", "frame_index"]
    if "prompt" in info.get("features", {}):
        columns.append("prompt")
    data = _load_parquet_columns(parquet_path, columns)

    actions = _stack_if_list(data["action"]).astype(np.float32)
    states = _stack_if_list(data["observation.state"]).astype(np.float32)
    frame_indices = np.asarray(data["frame_index"]).astype(int).reshape(-1)

    prompt = args.prompt
    if prompt is None and "prompt" in data:
        # prompt column may be a list of strings
        prompt_list = data["prompt"]
        prompt = prompt_list[0] if isinstance(prompt_list, list) else str(prompt_list[0])
    if prompt is None:
        raise ValueError("Prompt not found in parquet. Provide --prompt.")

    gripper = actions[:, args.gripper_index]
    close_indices = np.where(gripper < args.gripper_threshold)[0]
    if close_indices.size == 0:
        raise ValueError("No gripper-close event found with given threshold.")
    close_idx = int(close_indices[0])
    if close_idx == 0:
        raise ValueError("First frame already closed; no previous frame available.")

    prev_idx = close_idx - 1
    frame_index = int(frame_indices[prev_idx])

    video_path_fmt = info["video_path"]
    chunk_size = int(info["chunks_size"])
    episode_chunk = episode_index // chunk_size

    def video_path(video_key: str) -> Path:
        rel = video_path_fmt.format(
            episode_chunk=episode_chunk,
            video_key=video_key,
            episode_index=episode_index,
        )
        return dataset_root / rel

    image_front = _read_video_frame(video_path(args.image_front_key), frame_index)
    image_wrist = _read_video_frame(video_path(args.image_wrist_key), frame_index)

    observation = {
        "observation.image_front": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(image_front, 224, 224)
        ),
        "observation.image_wrist": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(image_wrist, 224, 224)
        ),
        "observation.state": states[prev_idx],
        "prompt": prompt,
    }

    if args.disable_ping:
        import websockets.sync.client

        uri = f"ws://{args.host}:{args.port}"
        conn = websockets.sync.client.connect(uri, compression=None, max_size=None, ping_interval=None)
        _ = msgpack_numpy.unpackb(conn.recv())  # server metadata
        conn.send(msgpack_numpy.Packer().pack(observation))
        response = msgpack_numpy.unpackb(conn.recv())
        conn.close()
    else:
        client = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
        response = client.infer(observation)
    pred_actions = response.get("actions")

    print("prev_idx:", prev_idx)
    print("close_idx:", close_idx)
    print("gt_action_next:", actions[close_idx])
    print("pred_actions:", pred_actions)


if __name__ == "__main__":
    main()
