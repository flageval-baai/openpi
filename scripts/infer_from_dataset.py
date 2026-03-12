"""Load one episode from a LeRobot dataset and run inference on all frames via websocket."""

import json
import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from openpi_client import websocket_client_policy

DATASET_ROOT = "/share/project/embodied/stage/data/processed/pose_euler/B1a_Put_the_building_blocks_into_the_basket_260305"
REPO_ID = "B1a_Put_the_building_blocks_into_the_basket_260305"
EPISODE_INDEX = 12
OUTPUT_PATH = "output/actions_from_dataset.json"
PROMPT = "Put the building blocks into the basket"

# Mapping from LeRobot dataset keys to websocket client keys
CAMERA_KEY_MAP = {
    "observation.images.image_front": "observation.image_front",
    "observation.images.image_wrist": "observation.image_wrist",
    "observation.images.image_side": "observation.image_side",
}


def tensor_image_to_uint8(img: torch.Tensor) -> np.ndarray:
    """Convert a CHW float32 [0,1] tensor to HWC uint8 numpy array."""
    img = img.numpy()
    if img.shape[0] == 3:  # CHW -> HWC
        img = np.transpose(img, (1, 2, 0))
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    else:
        img = img.astype(np.uint8)
    return img


def main():
    # 1. Load dataset
    dataset = LeRobotDataset(repo_id=REPO_ID, root=DATASET_ROOT, episodes=[EPISODE_INDEX])
    num_frames = len(dataset)
    print(f"Loaded dataset: {num_frames} frames, fps={dataset.fps}")

    # 2. Connect to websocket server
    policy = websocket_client_policy.WebsocketClientPolicy(
        host="localhost",
        port=8000,
    )
    print(f"Server metadata: {policy.get_server_metadata()}")

    # 3. Inference all frames
    output_frames = []
    for i in range(num_frames):
        sample = dataset[i]

        state = sample["observation.state"].numpy().astype(np.float32)

        obs = {
            "observation.state": state,
            "prompt": PROMPT,
        }
        for ds_key, client_key in CAMERA_KEY_MAP.items():
            obs[client_key] = tensor_image_to_uint8(sample[ds_key])
            # save images
            from PIL import Image
            img = Image.fromarray(obs[client_key])
            
            img.save(f"output/images/frame_{i}_{client_key.replace('observation.','')}.png")


        result = policy.infer(obs)
        first_action = result["actions"][0].tolist()

        output_frames.append({
            "index": i,
            "episode_index": int(sample["episode_index"]),
            "frame_index": int(sample["frame_index"]),
            "timestamp": float(sample["timestamp"]),
            "action": first_action,
            "observation.state": state.tolist(),
        })

        if (i + 1) % 50 == 0 or i == 0:
            print(f"  [{i + 1}/{num_frames}] frame_index={int(sample['frame_index'])}, action={first_action[:3]}...")

    # 4. Save results
    output_data = {
        "metadata": {
            "num_frames": num_frames,
            "num_episodes": 1,
            "fps": 30,
        },
        "frames": output_frames,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"Saved {num_frames} frames to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
