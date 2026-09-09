import os

# import torch
# import wandb
from crew_algorithms.envs.channels import ToggleTimestepChannel
from crew_algorithms.envs.configs import EnvironmentConfig

# from crew_algorithms.multimodal.split_key_transform import IndexSelectTransform
from crew_algorithms.utils.rl_utils import (
    # convert_tensor_to_pil_image,
    make_base_env,
    unsqueeze_images_from_channel_dimension,
)
# from torchrl.collectors.collectors import RandomPolicy
from torchrl.envs import (
    Compose,
    EnvBase,  # , StepCounter, ToTensorImage
    TransformedEnv,
)


def make_env(
    cfg: EnvironmentConfig,
    toggle_timestep_channel: ToggleTimestepChannel,
    device: str,
):

    env = TransformedEnv(
        make_base_env(cfg, device, toggle_timestep_channel=toggle_timestep_channel),
        Compose(
            # IndexSelectTransform([[torch.tensor([0])]], [[1]],
            # in_keys=[("agents", "observation", "obs_0")],
            # out_keys=[("agents", "observation", "feedback")])
            # ToTensorImage(in_keys=[("agents", "observation")], unsqueeze=True),
        ),
    )
    return env




