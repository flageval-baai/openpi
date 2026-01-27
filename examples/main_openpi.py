# ruff: noqa

import contextlib
import dataclasses
import datetime
import faulthandler
import os
import signal
import time
from moviepy.editor import ImageSequenceClip
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy
import pandas as pd
from PIL import Image
from droid.robot_env import RobotEnv
import tqdm
import tyro

# --- NEW: Import OpenCV for live image display ---
import cv2

faulthandler.enable()

# DROID data collection frequency -- we slow down execution to match this frequency
DROID_CONTROL_FREQUENCY = 15


@dataclasses.dataclass
class Args:
    # Hardware parameters
    left_camera_id: str = "21348267"
    right_camera_id: str = "29660059"
    wrist_camera_id: str = "12578374"

    # Policy parameters
    external_camera: str | None = "left"

    # Rollout parameters
    max_timesteps: int = 4000
    open_loop_horizon: int = 8

    # Remote server parameters
    remote_host: str = "0.0.0.0"
    remote_port: int = 8000


@contextlib.contextmanager
def prevent_keyboard_interrupt():
    interrupted = False
    original_handler = signal.getsignal(signal.SIGINT)

    def handler(signum, frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, original_handler)
        if interrupted:
            raise KeyboardInterrupt


def main(args: Args):
    assert (
        args.external_camera is not None and args.external_camera in ["left", "right"]
    ), f"Please specify an external camera to use for the policy, choose from ['left', 'right'], but got {args.external_camera}"

    env = RobotEnv(action_space="joint_velocity", gripper_action_space="position")
    print("Created the droid env!")

    policy_client = websocket_client_policy.WebsocketClientPolicy(args.remote_host, args.remote_port)
    df = pd.DataFrame(columns=["success", "duration", "video_filename"])
    os.makedirs("inference_logs", exist_ok=True)

    # --- MODIFICATION: Setup OpenCV windows and ensure cleanup ---
    # Create windows that can be resized
    cv2.namedWindow("Left Camera", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Right Camera", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Wrist Camera", cv2.WINDOW_NORMAL)
    
    try:
        while True:
            instruction = input("Enter instruction: ")

            timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            log_dir = os.path.join("inference_logs", f"rollout_{timestamp}")
            os.makedirs(log_dir, exist_ok=True)
            inference_count = 0
            print(f"Logging inference data for this rollout to: {log_dir}")

            actions_from_chunk_completed = 0
            pred_action_chunk = None
            video = []
            bar = tqdm.tqdm(range(args.max_timesteps))
            print("Running rollout... press Ctrl+C to stop early.")
            
            env.reset()

            for t_step in bar:
                start_time = time.time()
                try:
                    # Get the current observation. The save_to_disk flag is no longer needed.
                    curr_obs = _extract_observation(args, env.get_observation())

                    # --- MODIFICATION: Display images in OpenCV windows ---
                    # OpenCV expects images in BGR format, so we convert from RGB
                    left_bgr = curr_obs["left_image"]
                    right_bgr = curr_obs["right_image"]
                    wrist_bgr = curr_obs["wrist_image"]

                    cv2.imshow("Left Camera", left_bgr)
                    cv2.imshow("Right Camera", right_bgr)
                    cv2.imshow("Wrist Camera", wrist_bgr)
                    # --- END MODIFICATION ---

                    video.append(curr_obs[f"{args.external_camera}_image"])

                    if actions_from_chunk_completed == 0 or actions_from_chunk_completed >= args.open_loop_horizon:
                        actions_from_chunk_completed = 0
                        request_data = {
                            "observation.exterior_image_1_left": image_tools.resize_with_pad(
                                left_bgr, 224, 224
                            ),
                            "observation.wrist_image_left": image_tools.resize_with_pad(wrist_bgr, 224, 224),
                            "observation.joint_position": curr_obs["joint_position"],
                            "observation.gripper_position": curr_obs["gripper_position"],
                            "prompt": instruction,
                        }
                        with prevent_keyboard_interrupt():
                            pred_action_chunk = policy_client.infer(request_data)["actions"]

                        # The detailed logging for debugging remains unchanged
                        inference_step_dir = os.path.join(log_dir, f"inference_{inference_count:03d}")
                        os.makedirs(inference_step_dir, exist_ok=True)
                        with open(os.path.join(inference_step_dir, 'prompt.txt'), 'w') as f:
                            f.write(instruction)
                        Image.fromarray(curr_obs['wrist_image']).save(os.path.join(inference_step_dir, 'observation_wrist_image.png'))
                        Image.fromarray(curr_obs[f'{args.external_camera}_image']).save(os.path.join(inference_step_dir, 'observation_external_image.png'))
                        np.save(os.path.join(inference_step_dir, 'observation_joint_position.npy'), curr_obs['joint_position'])
                        np.save(os.path.join(inference_step_dir, 'observation_gripper_position.npy'), curr_obs['gripper_position'])
                        np.save(os.path.join(inference_step_dir, 'predicted_action_chunk.npy'), pred_action_chunk)
                        inference_count += 1

                    action = pred_action_chunk[:].copy()
                    actions_from_chunk_completed += 1
                    for act in action:
                        if act[-1].item() > 0.5:
                            act[-1] = 1.0
                        else:
                            act[-1] = 0.0
                    
                        env.step(act)

                        elapsed_time = time.time() - start_time
                        if elapsed_time < 1 / DROID_CONTROL_FREQUENCY:
                            time.sleep(1 / DROID_CONTROL_FREQUENCY - elapsed_time)
                        
                    # --- MODIFICATION: Add waitKey to update OpenCV windows ---
                    # This is crucial. It waits 1ms for a key press and allows OpenCV to process GUI events.
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("'q' pressed, stopping rollout.")
                        break
                    # --- END MODIFICATION ---

                except KeyboardInterrupt:
                    print("\nRollout interrupted by user.")
                    break

            if video:
                video = np.stack(video)
                save_filename = "video_" + timestamp
                ImageSequenceClip(list(video), fps=10).write_videofile(save_filename + ".mp4", codec="libx264", logger=None)
                print(f"\nSaved rollout video to: {save_filename}.mp4")
            else:
                save_filename = "no_video"
                print("\nNo frames recorded, skipping video saving.")

            success: str | float | None = None
            while not isinstance(success, float) or not (0 <= success <= 1):
                try:
                    success_input = input("Did the rollout succeed? (y/n, or 0-100): ")
                    if success_input.lower() == "y": success = 1.0
                    elif success_input.lower() == "n": success = 0.0
                    else: success = float(success_input) / 100.0
                    if not (0 <= success <= 1): print(f"Success must be in [0, 100].")
                except ValueError:
                    print("Invalid input.")
                    success = None

            df_new_row = pd.DataFrame([{"success": success, "duration": t_step + 1 if 't_step' in locals() else 0, "video_filename": save_filename}])
            df = pd.concat([df, df_new_row], ignore_index=True)

            if input("Do one more eval? (y/n) ").lower() != "y":
                break
    finally:
        # --- MODIFICATION: Ensure all OpenCV windows are closed on exit ---
        print("Closing OpenCV windows.")
        cv2.destroyAllWindows()
        # --- END MODIFICATION ---

    os.makedirs("results", exist_ok=True)
    csv_timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S")
    csv_filename = os.path.join("results", f"eval_{csv_timestamp}.csv")
    df.to_csv(csv_filename, index=False)
    print(f"Results saved to {csv_filename}")


def _extract_observation(args: Args, obs_dict):
    image_observations = obs_dict["image"]
    left_image, right_image, wrist_image = None, None, None
    for key in image_observations:
        if args.left_camera_id in key and "left" in key:
            left_image = image_observations[key]
        elif args.right_camera_id in key and "left" in key:
            right_image = image_observations[key]
        elif args.wrist_camera_id in key and "left" in key:
            wrist_image = image_observations[key]

    if left_image is None or right_image is None or wrist_image is None:
        found_cams = {'left': left_image is not None, 'right': right_image is not None, 'wrist': wrist_image is not None}
        missing_cams = [cam for cam, found in found_cams.items() if not found]
        raise RuntimeError(f"Could not find observations for camera(s): {', '.join(missing_cams)}. Please check camera IDs.")

    left_image = left_image[..., :3][..., ::-1]
    right_image = right_image[..., :3][..., ::-1]
    wrist_image = wrist_image[..., :3][..., ::-1]

    robot_state = obs_dict["robot_state"]
    cartesian_position = np.array(robot_state["cartesian_position"])
    joint_position = np.array(robot_state["joint_positions"])
    gripper_position = np.array([robot_state["gripper_position"]])

    # --- MODIFICATION: Removed the save_to_disk logic ---
    # Displaying is now handled in the main loop.
    # This function is now purely for data extraction and formatting.

    return {
        "left_image": left_image,
        "right_image": right_image,
        "wrist_image": wrist_image,
        "cartesian_position": cartesian_position,
        "joint_position": joint_position,
        "gripper_position": gripper_position,
    }


if __name__ == "__main__":
    args: Args = tyro.cli(Args)
    main(args)