import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    """Convert input to uint8 HWC format.

    Handles inputs that may be float in [0, 1] or [0, 255], and CHW layout.
    Avoids ambiguous array truth-value comparisons.
    """
    # Convert possible torch tensors or other array-likes to numpy first
    try:
        import torch  # type: ignore

        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
    except Exception:
        pass

    image = np.asarray(image)

    # If float, decide whether to scale based on max value
    if np.issubdtype(image.dtype, np.floating):
        max_val = float(np.max(image)) if image.size > 0 else 0.0
        if max_val <= 1.0 + 1e-6:
            image = (255.0 * image).astype(np.uint8)
        else:
            image = image.astype(np.uint8)

    # If CHW, convert to HWC
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class FEFrankaInputs(transforms.DataTransformFn):
    # Determines which model will be used.
    model_type: _model.ModelType
    # Optional indices to select/reorder action dimensions from the raw dataset action.
    # If None, actions are used as-is.
    action_indices: tuple[int, ...] | None = None

    def __call__(self, data: dict) -> dict:
        #gripper_pos = np.asarray(data["observation.gripper_position"])
        #if gripper_pos.ndim == 0:
            # Ensure gripper position is a 1D array, not a scalar, so we can concatenate with joint positions
            #gripper_pos = gripper_pos[np.newaxis]
        #state = np.concatenate([data["observation.joint_position"], gripper_pos])
        state = np.asarray(data['observation.state'])
        if self.action_indices is not None:
            state = state[..., list(self.action_indices)]
        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference
        front_image = _parse_image(data["observation.image_front"])
        wrist_image = _parse_image(data["observation.image_wrist"])
        if "observation.image_side" in data:
            side_image = _parse_image(data["observation.image_side"])
            images = (front_image, wrist_image, side_image)
            image_masks = (np.True_, np.True_, np.True_)
        else:
            side_image = None
            images = (front_image, wrist_image,np.zeros_like(front_image))
            image_masks = (np.True_, np.True_, np.False_)

        names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
        

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        if "actions" in data:
            raw_actions = np.asarray(data["actions"])
            if self.action_indices is None:
                inputs["actions"] = raw_actions
            else:
                inputs["actions"] = raw_actions[..., list(self.action_indices)]

        if "prompt" in data:
            if isinstance(data["prompt"], bytes):
                data["prompt"] = data["prompt"].decode("utf-8")
            inputs["prompt"] = data["prompt"]
            print(f'Using prompt: {inputs["prompt"]}')
        else:
            inputs['prompt'] = 'Put the block in the basket'
        return inputs


@dataclasses.dataclass(frozen=True)
class FEFrankaOutputs(transforms.DataTransformFn):
    # Number of action dimensions to return from the (zero-padded) model output.
    action_dim: int = 7

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :self.action_dim])}