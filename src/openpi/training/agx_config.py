"""AGX robot training configs (pi0 and pi05)."""

from openpi.training.config import (
    AssetsConfig,
    DataConfig,
    LeRobotAlohaDataConfig,
    TrainConfig,
)
import openpi.models.pi0_config as pi0_config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

# Common paths
_PI0_BASE = "/share/project/hezheqi/projects/openpi/weights/pi0_base/params"
_PI05_BASE = "/share/project/hezheqi/projects/openpi/weights/pi05_base/params"
_ASSETS_ROOT = "/share/project/hezheqi/projects/openpi/assets"
_DATA_ROOT = "/share/project/embodied/siyuan/agx/lerobot"

# Common repack transform for AGX (3-camera aloha setup)
_AGX_REPACK = _transforms.Group(
    inputs=[
        _transforms.RepackTransform(
            {
                "images": {
                    "cam_high": "observation.images.cam_high",
                    "cam_left_wrist": "observation.images.cam_left_wrist",
                    "cam_right_wrist": "observation.images.cam_right_wrist",
                },
                "state": "observation.state",
                "actions": "action",
                "prompt": "prompt",
            }
        )
    ]
)

# Same repack but without prompt (used by norm_instruct variant)
_AGX_REPACK_NO_PROMPT = _transforms.Group(
    inputs=[
        _transforms.RepackTransform(
            {
                "images": {
                    "cam_high": "observation.images.cam_high",
                    "cam_left_wrist": "observation.images.cam_left_wrist",
                    "cam_right_wrist": "observation.images.cam_right_wrist",
                },
                "state": "observation.state",
                "actions": "action",
            }
        )
    ]
)

# Common LR schedule
_AGX_LR = _optimizer.CosineDecaySchedule(
    warmup_steps=200,
    peak_lr=2e-5,
    decay_steps=4000,
    decay_lr=1e-6,
)


def _pi0_agx_config(
    name: str,
    task_name: str,
    assets_name: str | None = None,
    batch_size: int = 16,
    save_interval: int = 500,
    keep_period: int = 1000,
) -> TrainConfig:
    """Create a pi0 AGX config.

    Args:
        task_name: Short task name (e.g. "erase_all"), used as subdirectory under _DATA_ROOT.
    """
    assets_name = assets_name or name
    return TrainConfig(
        name=name,
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id=f"{_DATA_ROOT}/{task_name}",
            assets=AssetsConfig(
                assets_dir=f"{_ASSETS_ROOT}/{assets_name}",
                asset_id=task_name,
            ),
            repack_transforms=_AGX_REPACK,
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(_PI0_BASE),
        lr_schedule=_AGX_LR,
        num_train_steps=4000,
        batch_size=batch_size,
        log_interval=100,
        save_interval=save_interval,
        keep_period=keep_period,
    )


def _pi05_agx_config(
    name: str,
    task_name: str,
    assets_name: str | None = None,
    batch_size: int = 16,
    save_interval: int = 500,
    keep_period: int = 1000,
) -> TrainConfig:
    """Create a pi05 AGX config.

    Args:
        task_name: Short task name (e.g. "erase_all"), used as subdirectory under _DATA_ROOT.
    """
    assets_name = assets_name or name
    return TrainConfig(
        name=name,
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
        ),
        data=LeRobotAlohaDataConfig(
            repo_id=f"{_DATA_ROOT}/{task_name}",
            assets=AssetsConfig(
                assets_dir=f"{_ASSETS_ROOT}/{assets_name}",
                asset_id=task_name,
            ),
            repack_transforms=_AGX_REPACK,
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(_PI05_BASE),
        lr_schedule=_AGX_LR,
        num_train_steps=4000,
        batch_size=batch_size,
        log_interval=100,
        save_interval=save_interval,
        keep_period=keep_period,
    )


def get_agx_configs() -> list[TrainConfig]:
    """Return all AGX training configs."""
    return [
        # ---- pi0 ----
        _pi0_agx_config(
            "pi0_agx_erase_all", "erase_all",
            assets_name="pi05_agx_erase_all",
            save_interval=2000, keep_period=2000,
        ),
        _pi0_agx_config(
            "pi0_agx_put_mouse_on_pad", "put_mouse_on_pad",
            assets_name="pi05_agx_put_mouse_on_pad",
        ),
        _pi0_agx_config(
            "pi0_agx_fruit_basket_cup_pad", "fruit_basket_cup_pad",
            assets_name="pi05_agx_fruit_basket_cup_pad",
        ),
        _pi0_agx_config(
            "pi0_agx_place_2nd_to_basket", "place_2nd_to_basket",
            assets_name="pi05_agx_place_2nd_to_basket",
            save_interval=2000, keep_period=2000,
        ),

        # ---- pi05 ----
        _pi05_agx_config(
            "pi05_agx_fruit_basket_cup_pad", "fruit_basket_cup_pad",
        ),
        _pi05_agx_config(
            "pi05_agx_erase_left", "erase_left",
        ),
        _pi05_agx_config(
            "pi05_agx_put_mouse_on_pad", "put_mouse_on_pad",
            save_interval=2000, keep_period=2000,
        ),
        _pi05_agx_config(
            "pi05_agx_place_2nd_to_basket", "place_2nd_to_basket",
            save_interval=2000, keep_period=2000,
        ),
        _pi05_agx_config(
            "pi05_agx_place_obj_closest_apple", "place_obj_closest_apple",
            save_interval=2000, keep_period=2000,
        ),
        _pi05_agx_config(
            "pi05_agx_pour_water_into_bottle_easy", "pour_water_into_bottle_easy",
        ),
        _pi05_agx_config(
            "pi05_agx_operate_microwave_easy", "operate_microwave_easy",
        ),
        _pi05_agx_config(
            "pi05_agx_erase_all", "erase_all",
        ),
        _pi05_agx_config(
            "pi05_agx_fold_towel_easy", "fold_towel_easy",
        ),
        _pi05_agx_config(
            "pi05_agx_close_drawer_hard", "close_drawer_hard",
        ),
        _pi05_agx_config(
            "pi05_agx_close_drawer_easy", "close_drawer_easy",
        ),
        _pi05_agx_config(
            "pi05_agx_operate_microwave_hard", "operate_microwave_hard",
        ),
        _pi05_agx_config(
            "pi05_agx_pour_water_into_bottle_hard", "pour_water_into_bottle_hard",
        ),
        _pi05_agx_config(
            "pi05_agx_open_drawer_easy", "open_drawer_easy",
        ),

        # Special: uses default_prompt and no prompt in repack
        TrainConfig(
            name="pi05_agx_put_mouse_on_pad_norm_instruct",
            model=pi0_config.Pi0Config(
                pi05=True,
                action_dim=32,
                action_horizon=16,
            ),
            data=LeRobotAlohaDataConfig(
                repo_id=f"{_DATA_ROOT}/put_mouse_on_pad",
                assets=AssetsConfig(
                    assets_dir=f"{_ASSETS_ROOT}/pi05_agx_put_mouse_on_pad",
                    asset_id="put_mouse_on_pad",
                ),
                default_prompt="put the mouse on the mousepad",
                repack_transforms=_AGX_REPACK_NO_PROMPT,
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(_PI05_BASE),
            lr_schedule=_AGX_LR,
            num_train_steps=4000,
            batch_size=16,
            log_interval=100,
            save_interval=500,
            keep_period=1000,
        ),
    ]
