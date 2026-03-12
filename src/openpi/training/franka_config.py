"""Franka robot training configs (pi05) and FrankaDataConfig."""

import dataclasses
import pathlib

from typing_extensions import override

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.policies.fe_franka_policy as fe_franka_policy
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms
from openpi.training.config import (
    AssetsConfig,
    DataConfig,
    DataConfigFactory,
    ModelTransformFactory,
    TrainConfig,
)

@dataclasses.dataclass(frozen=True)
class FrankaDataConfig(DataConfigFactory):
    """Data config for Franka robot datasets in LeRobot format."""

    default_prompt: str | None = None
    # Optional indices to select/reorder action dimensions from the raw dataset action.
    # If None, actions are used as-is.
    action_indices: tuple[int, ...] | None = None
    # Whether to convert absolute actions to delta actions (action - state) during training.
    # The inverse transform (delta -> absolute) is applied during inference.
    use_delta_actions: bool = False
    # Delta action mask: how many dims are delta (positive) vs absolute (negative).
    # Default: first 6 delta, last 1 absolute = (6, -1) for EEF [x,y,z,r,p,y,gripper].
    # For joint pos: (7, -1) means 7 joints delta, gripper absolute.
    delta_mask_spec: tuple[int, ...] = (6, -1)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation.image_front": "observation.images.image_front",
                        "observation.image_wrist": "observation.images.image_wrist",
                        "observation.image_side": "observation.images.image_side",
                        "observation.state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )
        # Determine output action dim from action_indices (or default 7)
        output_action_dim = len(self.action_indices) if self.action_indices is not None else 7
        data_transforms = _transforms.Group(
            inputs=[
                fe_franka_policy.FEFrankaInputs(
                    model_type=model_config.model_type,
                    action_indices=self.action_indices,
                )
            ],
            outputs=[fe_franka_policy.FEFrankaOutputs(action_dim=output_action_dim)],
        )

        if self.use_delta_actions:
            delta_action_mask = _transforms.make_bool_mask(*self.delta_mask_spec)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=("action",),
        )


# Common paths
_PI05_BASE = "/share/project/hezheqi/projects/openpi/weights/pi05_base/params"
_ASSETS_ROOT = "/share/project/hezheqi/projects/openpi/assets"
_B1A_DATASET = "/share/project/embodied/stage/data/processed/pose_euler/B1a_Put_the_building_blocks_into_the_basket_260305"

# EEF pose action indices: [x, y, z, roll, pitch, yaw, gripper] from 14D state
_EEF_ACTION_INDICES = (8, 9, 10, 11, 12, 13, 7)
# Joint position action indices: [j0..j6, gripper] from 14D state
_JOINT_ACTION_INDICES = (0, 1, 2, 3, 4, 5, 6, 7)


def get_franka_configs() -> list[TrainConfig]:
    """Return all Franka training configs."""
    return [
        # ---------- EEF pose, full fine-tuning ----------
        TrainConfig(
            name="pi05_franka_b1a",
            model=pi0_config.Pi0Config(
                pi05=True,
                action_dim=32,
                action_horizon=16,
                discrete_state_input=False,
            ),
            data=FrankaDataConfig(
                repo_id=_B1A_DATASET,
                action_indices=_EEF_ACTION_INDICES,
                base_config=DataConfig(prompt_from_task=True),
                default_prompt="Put the building blocks into the basket.",
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(_PI05_BASE),
            num_train_steps=4000,
            batch_size=32,
            log_interval=100,
            save_interval=500,
            keep_period=1000,
        ),

        # ---------- EEF pose, LoRA ----------
        TrainConfig(
            name="pi05_franka_b1a_lora",
            model=pi0_config.Pi0Config(
                pi05=True,
                action_dim=32,
                action_horizon=16,
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
            ),
            data=FrankaDataConfig(
                repo_id=_B1A_DATASET,
                action_indices=_EEF_ACTION_INDICES,
                base_config=DataConfig(prompt_from_task=True),
                default_prompt="Put the building blocks into the basket.",
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(_PI05_BASE),
            num_train_steps=5000,
            batch_size=32,
            freeze_filter=pi0_config.Pi0Config(
                pi05=True, action_dim=32, action_horizon=16,
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
            ).get_freeze_filter(),
            ema_decay=None,
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=200, peak_lr=5e-5, decay_steps=5000, decay_lr=1e-6,
            ),
            log_interval=100,
            save_interval=500,
            keep_period=1000,
        ),

        # ---------- EEF pose, full fine-tuning + delta actions (action_dim=7) ----------
        TrainConfig(
            name="pi05_franka_b1a_delta",
            model=pi0_config.Pi0Config(
                pi05=True,
                action_dim=7,
                action_horizon=16,
            ),
            data=FrankaDataConfig(
                repo_id=_B1A_DATASET,
                action_indices=_EEF_ACTION_INDICES,
                base_config=DataConfig(prompt_from_task=True),
                default_prompt="Put the building blocks into the basket.",
                use_delta_actions=True,
                assets=AssetsConfig(
                    assets_dir=f"{_ASSETS_ROOT}/pi05_franka_b1a_delta",
                    asset_id="B1a_Put_the_building_blocks_into_the_basket_260305",
                ),
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(_PI05_BASE),
            num_train_steps=4000,
            batch_size=32,
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=200, peak_lr=2e-5, decay_steps=4000, decay_lr=1e-6,
            ),
            ema_decay=0.999,
            log_interval=100,
            save_interval=500,
            keep_period=1000,
        ),

        # ---------- Joint position, full fine-tuning ----------
        TrainConfig(
            name="pi05_franka_b1a_joint",
            model=pi0_config.Pi0Config(
                pi05=True,
                action_dim=32,
                action_horizon=16,
            ),
            data=FrankaDataConfig(
                repo_id=_B1A_DATASET,
                action_indices=_JOINT_ACTION_INDICES,
                base_config=DataConfig(prompt_from_task=True),
                default_prompt="Put the building blocks into the basket.",
                assets=AssetsConfig(
                    assets_dir=f"{_ASSETS_ROOT}/pi05_franka_b1a_joint",
                    asset_id="B1a_Put_the_building_blocks_into_the_basket_260305",
                ),
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(_PI05_BASE),
            num_train_steps=4000,
            batch_size=32,
            log_interval=100,
            save_interval=500,
            keep_period=1000,
        ),

        # ----------with eval: EEF pose, full fine-tuning ----------
        TrainConfig(
            name="pi05_franka_b1a_with_eval",
            model=pi0_config.Pi0Config(
                pi05=True,
                action_dim=32,
                action_horizon=16,
                discrete_state_input=False,
            ),
            data=FrankaDataConfig(
                repo_id=_B1A_DATASET,
                action_indices=_EEF_ACTION_INDICES,
                base_config=DataConfig(
                    prompt_from_task=True,
                    episodes=list(range(80)),
                ),
            ),
            val_data=FrankaDataConfig(
                repo_id=_B1A_DATASET,
                base_config=DataConfig(
                    prompt_from_task=True,
                    episodes=list(range(80, 100)),
                ),
            ),
            val_interval=500,
            weight_loader=weight_loaders.CheckpointWeightLoader(_PI05_BASE),
            num_train_steps=4000,
            batch_size=32,
            log_interval=100,
            save_interval=500,
            keep_period=1000,
        ),
    ]
