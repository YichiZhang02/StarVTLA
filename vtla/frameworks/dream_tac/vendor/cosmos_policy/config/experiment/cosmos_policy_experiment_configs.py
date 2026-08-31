# -----------------------------------------------------------------------------
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This codebase constitutes NVIDIA proprietary technology and is strictly
# confidential. Any unauthorized reproduction, distribution, or disclosure
# of this code, in whole or in part, outside NVIDIA is strictly prohibited
# without prior written consent.
#
# For inquiries regarding the use of this code in other NVIDIA proprietary
# projects, please contact the Deep Imagination Research Team at
# dir@exchange.nvidia.com.
# -----------------------------------------------------------------------------

# Modified by StarVTLA: resolve the Cosmos base checkpoint through
# COSMOS_POLICY_BASE_CHECKPOINT instead of a machine-specific path.

import os

from hydra.core.config_store import ConfigStore
from megatron.core import parallel_state
from torch.utils.data import DataLoader, DistributedSampler

from cosmos_policy._src.imaginaire.lazy_config import LazyCall as L
from cosmos_policy._src.imaginaire.lazy_config import LazyDict
from cosmos_policy._src.imaginaire.utils import log
from cosmos_policy._src.imaginaire.utils.checkpoint_db import get_checkpoint_path  # noqa: F401
from cosmos_policy.datasets.aloha_dataset import ALOHADataset
from cosmos_policy.datasets.franka_dataset import FrankaDataset
from cosmos_policy.datasets.libero_dataset import LIBERODataset
from cosmos_policy.datasets.robocasa_dataset import RoboCasaDataset
from cosmos_policy.models.policy_video2world_model import CosmosPolicyVideo2WorldModel
from cosmos_policy.modules.hybrid_edm_sde import HybridEDMSDE

cs = ConfigStore.instance()
val_sampling_size_override = dict(
    video_length=121,
    video_height=704,
    video_width=1280,
)
BASE_DATASETS_DIR = os.environ.get("BASE_DATASETS_DIR", ".")
BASE_CHECKPOINT_PATH = os.environ.get(
    "COSMOS_POLICY_BASE_CHECKPOINT",
    "/path/to/Cosmos-Predict2-2B-Video2World/model-480p-16fps.pt",
)


# *** Main checkpoint ***
libero_all_4_suites_dataset = L(LIBERODataset)(
    data_dir=os.path.join(BASE_DATASETS_DIR, "LIBERO-Cosmos-Policy", "success_only"),  # Successful demos
    t5_text_embeddings_path=os.path.join(
        BASE_DATASETS_DIR, "LIBERO-Cosmos-Policy", "success_only", "t5_embeddings.pkl"
    ),
    chunk_size=16,
    use_image_aug=True,
    use_wrist_images=True,
    use_proprio=True,
    normalize_proprio=True,
    normalize_actions=True,
    num_duplicates_per_image=4,  # WAN 2.1 tokenizer: 4 images per latent frame
    use_stronger_image_aug=True,
    rollout_data_dir=os.path.join(
        BASE_DATASETS_DIR, "LIBERO-Cosmos-Policy", "all_episodes"
    ),  # All demo rollouts (successes + failures)
    demonstration_sampling_prob=0.5,
    success_rollout_sampling_prob=0.5,
    return_value_function_returns=True,
    gamma=0.99,
)
cosmos_predict2_2b_480p_libero = LazyDict(
    dict(
        defaults=[
            "/experiment/Stage-c_pt_4-Index-102-Size-2B-Res-480-Fps-16-Note-HQ_V5_from_26",
            {"override /data_train": "mock"},
            {"override /model": "policy_fsdp"},
            {"override /tokenizer": "policy_wan2pt1_tokenizer"},
            {
                "override /callbacks": [
                    "basic",
                    "long",
                    "cluster_speed",
                    "wandb",
                    "wandb_callback_actions",
                ]
            },
            "_self_",
        ],
        trainer=dict(
            callbacks=dict(
                every_n_sample_reg=dict(
                    every_n=100000,
                    save_s3=False,
                    use_negative_prompt=False,
                    guidance=[0],
                    num_sampling_step=9,
                ),
            ),
            run_validation=False,
            logging_iter=5,
            max_iter=1000000,
            straggler_detection=dict(
                enabled=False,
            ),
        ),
        optimizer=dict(
            lr=1e-4,
        ),
        scheduler=dict(
            # LR decay for 30K steps in cycle #1, then decay by 5x and stay constant forever in cycle #2
            cycle_lengths=[30000, 100000000000000],
            warm_up_steps=[1000, 0],
            f_start=[1e-6, 0.06],
            f_max=[1.0, 0.06],
            f_min=[0.3, 0.06],
        ),
        model=L(CosmosPolicyVideo2WorldModel)(
            config=dict(
                conditioner=dict(
                    text=dict(
                        # IMPORTANT: We don't want any text dropout; otherwise, the model may fail to follow language
                        dropout_rate=0.0,
                    ),
                ),
                state_t=8,  # Latent temporal dim (blank, proprio, wrist, primary, action, future proprio, future wrist, future primary, value)
                min_num_conditional_frames=4,  # 1 blank, 3 conditioning (proprio, wrist, primary)
                max_num_conditional_frames=4,  # 1 blank, 3 conditioning (proprio, wrist, primary)
                sigma_conditional=0.0,  # No noise on conditional latents
                conditioning_strategy="frame_replace",
                denoise_replace_gt_frames=True,
                tokenizer=dict(
                    chunk_duration=29,  # 1 blank + 32 images (4 proprio, 4 wrist image, 4 primary image, 4 action, 4 future proprio, 4 future wrist, 4 future primary, 4 value)
                ),
                ema=dict(
                    enabled=False,
                ),
                input_data_key="video",
                sde=L(HybridEDMSDE)(
                    hybrid_sigma_distribution=True,
                    p_mean=1.3862943611198906,  # Copied from base model config
                    p_std=1.2,
                    sigma_max=200,
                    sigma_min=0.01,
                    uniform_lower=1.0,
                    uniform_upper=85.0,
                ),
                adjust_video_noise=True,
                resize_online=True,
                resolution="224",
                high_sigma_strategy="none",
            ),
        ),
        model_parallel=dict(
            context_parallel_size=1,
        ),
        checkpoint=dict(
            load_path=get_checkpoint_path(BASE_CHECKPOINT_PATH),
            load_training_state=False,  # This means do not load train state from the base checkpoint above (load_path); but when resuming this job, will load train state
            strict_resume=False,
            save_iter=1000,
            load_ema_to_reg=True,
            load_from_object_store=dict(
                enabled=False,
            ),
            save_to_object_store=dict(
                enabled=False,
            ),
        ),
        dataloader_train=L(DataLoader)(
            num_workers=12,
            persistent_workers=True,
            pin_memory=True,
            dataset=libero_all_4_suites_dataset,
            sampler=L(DistributedSampler)(
                dataset=libero_all_4_suites_dataset,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=30,
            drop_last=True,
        ),
        job=dict(
            group="cosmos_v2_finetune",
            name="cosmos_predict2_2b_480p_libero",
        ),
        upload_reproducible_setup=False,
    )
)
# Inference version
cosmos_predict2_2b_480p_libero__inference_only = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_libero",
            "_self_",
        ],
        model=L(CosmosPolicyVideo2WorldModel)(
            config=dict(
                sde=L(HybridEDMSDE)(
                    sigma_max=80,
                    sigma_min=4,
                )
            )
        ),
        job=dict(
            group="cosmos_v2_inference",
            name="cosmos_predict2_2b_480p_libero__inference_only",
        ),
    )
)


# *** Main checkpoint ***
robocasa_50_demos_per_task_dataset = L(RoboCasaDataset)(
    data_dir=os.path.join(BASE_DATASETS_DIR, "RoboCasa-Cosmos-Policy", "success_only"),  # Successful demos
    t5_text_embeddings_path=os.path.join(
        BASE_DATASETS_DIR, "RoboCasa-Cosmos-Policy", "success_only", "t5_embeddings.pkl"
    ),
    chunk_size=32,
    use_image_aug=True,
    use_wrist_images=True,
    use_third_person_images=True,
    use_proprio=True,
    normalize_proprio=True,
    normalize_actions=True,
    num_duplicates_per_image=4,  # WAN 2.1 tokenizer: 4 images per latent frame
    use_stronger_image_aug=True,
    rollout_data_dir=os.path.join(
        BASE_DATASETS_DIR, "RoboCasa-Cosmos-Policy", "all_episodes"
    ),  # All demo rollouts (successes + failures)
    demonstration_sampling_prob=0.5,
    success_rollout_sampling_prob=0.5,
    return_value_function_returns=True,
    gamma=0.99,
)
cosmos_predict2_2b_480p_robocasa_50_demos_per_task = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_libero",
            "_self_",
        ],
        model=L(CosmosPolicyVideo2WorldModel)(
            config=dict(
                state_t=11,  # Latent temporal dim (blank, proprio, wrist image, primary image, secondary image, action, future proprio, future wrist image, future primary image, future secondary image, value)
                min_num_conditional_frames=5,  # 1 blank, 4 conditioning (proprio, wrist image, primary image, secondary image)
                max_num_conditional_frames=5,  # 1 blank, 4 conditioning (proprio, wrist image, primary image, secondary image)
                tokenizer=dict(
                    chunk_duration=41,  # 1 blank + 40 images (4 proprio, 4 wrist image, 4 primary image, 4 secondary image, 4 action, 4 future proprio, 4 future wrist, 4 future primary, 4 future secondary, 4 value)
                ),
            ),
        ),
        dataloader_train=L(DataLoader)(
            num_workers=8,
            persistent_workers=True,
            pin_memory=True,
            dataset=robocasa_50_demos_per_task_dataset,
            sampler=L(DistributedSampler)(
                dataset=robocasa_50_demos_per_task_dataset,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=25,
            drop_last=True,
        ),
        job=dict(
            group="cosmos_v2_finetune",
            name="cosmos_predict2_2b_480p_robocasa_50_demos_per_task",
        ),
    )
)
# Inference version
cosmos_predict2_2b_480p_robocasa_50_demos_per_task__inference = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_robocasa_50_demos_per_task",
            "_self_",
        ],
        model=L(CosmosPolicyVideo2WorldModel)(
            config=dict(
                sde=L(HybridEDMSDE)(
                    sigma_max=80,
                    sigma_min=4,
                )
            )
        ),
        job=dict(
            group="cosmos_v2_inference",
            name="cosmos_predict2_2b_480p_robocasa_50_demos_per_task__inference",
        ),
    )
)


# *** Main checkpoint ***
aloha_cosmos_policy_dataset_185_demos = L(ALOHADataset)(
    data_dir=os.path.join(BASE_DATASETS_DIR, "ALOHA-Cosmos-Policy", "preprocessed"),
    t5_text_embeddings_path=os.path.join(BASE_DATASETS_DIR, "ALOHA-Cosmos-Policy", "preprocessed", "t5_embeddings.pkl"),
    chunk_size=50,
    use_image_aug=True,
    use_stronger_image_aug=True,
    use_proprio=True,
    normalize_proprio=True,
    normalize_actions=True,
    num_duplicates_per_image=4,  # WAN 2.1 tokenizer: 4 images per latent frame
    treat_demos_as_success_rollouts=True,  # Include demos as success rollouts
    demonstration_sampling_prob=0.5,
    success_rollout_sampling_prob=0.5,
    return_value_function_returns=True,
    gamma=0.998,  # Higher gamma for ALOHA because episodes can have up to 1.5-2.0K steps  # (s, a, s', v)
)

# Franka/tactile: single-arm, cam_front=主视角, cam_high=腕部 (inherits LIBERODataset)
franka_pick_and_place_baguette_dataset = L(FrankaDataset)(
    data_dir="/path/to/tactile_data_final_action/pick_and_place_baguette_from_plate_to_basket",
    t5_text_embeddings_path="/path/to/tactile_data_final_action/pick_and_place_baguette_from_plate_to_basket/t5_embeddings.pkl",
    chunk_size=20,
    use_image_aug=True,
    use_stronger_image_aug=True,
    use_wrist_images=True,
    use_third_person_images=True,
    use_proprio=True,
    normalize_proprio=True,
    normalize_actions=True,
    num_duplicates_per_image=4,
    demonstration_sampling_prob=1.0,
    success_rollout_sampling_prob=0.0,
    return_value_function_returns=False,
    gamma=0.99,
)

# Franka whiteboard: state 6-dim (pose only), action 7-dim (pose + gripper). Data under tactile_data_preprocessed_gripper.
# Run once to generate T5 embeddings: uv run -m cosmos_policy.datasets.save_franka_t5_text_embeddings --data_dir /path/to/tactile_data_preprocessed_gripper/pick_eraser_and_erase_marker_from_whiteboard
franka_whiteboard_dataset = L(FrankaDataset)(
    data_dir="/path/to/tactile_data_preprocessed_gripper/pick_eraser_and_erase_marker_from_whiteboard",
    t5_text_embeddings_path="/path/to/tactile_data_preprocessed_gripper/pick_eraser_and_erase_marker_from_whiteboard/t5_embeddings.pkl",
    chunk_size=20,
    use_image_aug=True,
    use_stronger_image_aug=True,
    use_wrist_images=True,
    use_third_person_images=True,
    use_proprio=True,
    normalize_proprio=True,
    normalize_actions=True,
    num_duplicates_per_image=4,
    demonstration_sampling_prob=1.0,
    success_rollout_sampling_prob=0.0,
    return_value_function_returns=False,
    gamma=0.99,
)

cosmos_predict2_2b_480p_aloha_185_demos_4_tasks_mixture_foldshirt15_candiesinbowl45_candyinbag45_eggplantchickenonplate80 = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_libero",
            "_self_",
        ],
        scheduler=dict(
            # LR decay for 20K steps in cycle #1, then decay by 5x and stay constant forever in cycle #2
            cycle_lengths=[20000, 100000000000000],
            warm_up_steps=[2000, 0],
            f_start=[1e-6, 0.06],
            f_max=[1.0, 0.06],
            f_min=[0.3, 0.06],
        ),
        model=L(CosmosPolicyVideo2WorldModel)(
            config=dict(
                state_t=8,  # Latent temporal dim (blank, proprio, left wrist, right wrist, primary, action, future proprio, future left wrist, future right wrist, future primary, value)
                min_num_conditional_frames=4,  # 1 blank, 4 conditioning (proprio, left wrist, right wrist, primary)
                max_num_conditional_frames=4,  # 1 blank, 4 conditioning (proprio, left wrist, right wrist, primary)
                tokenizer=dict(
                    chunk_duration=29,  # 1 blank + 40 images (4 proprio, 4 left wrist image, 4 right wrist image, 4 primary image, 4 action, 4 future proprio, 4 future left wrist, 4 future right wrist, 4 future primary, 4 value)
                ),
            ),
        ),
        dataloader_train=L(DataLoader)(
            num_workers=12,
            persistent_workers=True,
            pin_memory=True,
            dataset=aloha_cosmos_policy_dataset_185_demos,
            sampler=L(DistributedSampler)(
                dataset=aloha_cosmos_policy_dataset_185_demos,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=25,
            drop_last=True,
        ),
        job=dict(
            group="cosmos_v2_finetune",
            name="cosmos_predict2_2b_480p_aloha_185_demos_4_tasks_mixture_foldshirt15_candiesinbowl45_candyinbag45_eggplantchickenonplate80",
        ),
    )
)
# Inference version
cosmos_predict2_2b_480p_aloha_185_demos_4_tasks_mixture_foldshirt15_candiesinbowl45_candyinbag45_eggplantchickenonplate80__inference_only = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_aloha_185_demos_4_tasks_mixture_foldshirt15_candiesinbowl45_candyinbag45_eggplantchickenonplate80",
            "_self_",
        ],
        model=L(CosmosPolicyVideo2WorldModel)(
            config=dict(
                sde=L(HybridEDMSDE)(
                    sigma_max=80,
                    sigma_min=4,
                )
            )
        ),
        job=dict(
            group="cosmos_v2_inference",
            name="cosmos_predict2_2b_480p_aloha_185_demos_4_tasks_mixture_foldshirt15_candiesinbowl45_candyinbag45_eggplantchickenonplate80__inference_only",
        ),
    )
)

# Franka: single-arm, cam_front=主视角, cam_high=腕部 (inherits LIBERO model: state_t=9, 2 views)
cosmos_predict2_2b_480p_franka_pick_and_place_baguette = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_libero",
            "_self_",
        ],
        scheduler=dict(
            cycle_lengths=[20000, 100000000000000],
            warm_up_steps=[2000, 0],
            f_start=[1e-6, 0.06],
            f_max=[1.0, 0.06],
            f_min=[0.3, 0.06],
        ),
        dataloader_train=L(DataLoader)(
            num_workers=12,
            persistent_workers=True,
            pin_memory=True,
            dataset=franka_pick_and_place_baguette_dataset,
            sampler=L(DistributedSampler)(
                dataset=franka_pick_and_place_baguette_dataset,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=25,
            drop_last=True,
        ),
        job=dict(
            project="cosmos_policy_franka",  # wandb project name (change as needed)
            group="cosmos_v2_finetune",
            name="cosmos_predict2_2b_480p_franka_pick_and_place_baguette",
            wandb_id="franka_pick_and_place_baguette",  # fixed run id: repeated runs log to the same wandb run
        ),
    )
)

# Franka whiteboard: state 6-dim, action 7-dim. Same model layout as baguette, different data.
cosmos_predict2_2b_480p_franka_whiteboard = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_libero",
            "_self_",
        ],
        scheduler=dict(
            cycle_lengths=[20000, 100000000000000],
            warm_up_steps=[2000, 0],
            f_start=[1e-6, 0.06],
            f_max=[1.0, 0.06],
            f_min=[0.3, 0.06],
        ),
        dataloader_train=L(DataLoader)(
            num_workers=12,
            persistent_workers=True,
            pin_memory=True,
            dataset=franka_whiteboard_dataset,
            sampler=L(DistributedSampler)(
                dataset=franka_whiteboard_dataset,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=25,
            drop_last=True,
        ),
        job=dict(
            project="cosmos_policy_franka_whiteboard",
            group="cosmos_v2_finetune",
            name="cosmos_predict2_2b_480p_franka_whiteboard",
            wandb_id="franka_whiteboard",
        ),
    )
)

# Franka hupai with tactile: state 6D, action 7D, + tactile_rectify_left/right as input & prediction target.
# Sequence (12 slots): blank, proprio, wrist, primary, tactile_left, tactile_right,
#                       action, future_proprio, future_wrist, future_primary, future_tactile_left, future_tactile_right
# Run once to generate T5 embeddings: uv run -m cosmos_policy.datasets.save_franka_t5_text_embeddings --data_dir /path/to/tactile_data_hupai/hupai
franka_hupai_tactile_dataset = L(FrankaDataset)(
    data_dir="/path/to/tactile_data_hupai/hupai",
    t5_text_embeddings_path="/path/to/tactile_data_hupai/hupai/t5_embeddings.pkl",
    chunk_size=20,
    use_image_aug=True,
    use_stronger_image_aug=True,
    use_wrist_images=True,
    use_third_person_images=True,
    use_proprio=True,
    normalize_proprio=True,
    normalize_actions=True,
    num_duplicates_per_image=4,
    demonstration_sampling_prob=1.0,
    success_rollout_sampling_prob=0.0,
    return_value_function_returns=False,
    gamma=0.99,
)

cosmos_predict2_2b_480p_franka_hupai_tactile = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_libero",
            "_self_",
        ],
        scheduler=dict(
            cycle_lengths=[20000, 100000000000000],
            warm_up_steps=[2000, 0],
            f_start=[1e-6, 0.06],
            f_max=[1.0, 0.06],
            f_min=[0.3, 0.06],
        ),
        model=L(CosmosPolicyVideo2WorldModel)(
            config=dict(
                state_t=12,  # 12 latent slots (see sequence above)
                min_num_conditional_frames=6,  # 1 blank + 5 conditioning (proprio, wrist, primary, tactile_left, tactile_right)
                max_num_conditional_frames=6,
                tokenizer=dict(
                    chunk_duration=45,  # get_pixel_num_frames(12) = (12-1)*4 + 1 = 45
                ),
                # Non-tactile Q -> tactile K logit bias: α * g * (1-M_i) * M_j (see MiniTrainDIT)
                net=dict(
                    use_tactile_self_attn_bias=True,
                    tactile_self_attn_alpha=2.0,
                    tactile_latent_t_indices=(4, 5, 10, 11),
                    tactile_attn_chunk_q=32,
                ),
            ),
        ),
        dataloader_train=L(DataLoader)(
            num_workers=12,
            persistent_workers=True,
            pin_memory=True,
            dataset=franka_hupai_tactile_dataset,
            sampler=L(DistributedSampler)(
                dataset=franka_hupai_tactile_dataset,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=16,
            drop_last=True,
        ),
        checkpoint=dict(
            save_iter=50,
        ),
        job=dict(
            project="cosmos_policy_franka_hupai",
            group="cosmos_v2_finetune",
            name="cosmos_predict2_2b_480p_franka_hupai_tactile",
            wandb_id="franka_hupai_tactile",
        ),
    )
)

# Franka shave cucumber (tactile): preprocessed with preprocess_tactile_franka_data.py; T5 via save_franka_t5_text_embeddings.
_SHAVE_CUCUMBER_20260321_DIR = "/path/to/attention_data_shave_cucumber/shave_cucumber_20260321"
franka_shave_cucumber_20260321_dataset = L(FrankaDataset)(
    data_dir=_SHAVE_CUCUMBER_20260321_DIR,
    t5_text_embeddings_path=os.path.join(_SHAVE_CUCUMBER_20260321_DIR, "t5_embeddings.pkl"),
    chunk_size=20,
    use_image_aug=True,
    use_stronger_image_aug=True,
    use_wrist_images=True,
    use_third_person_images=True,
    use_proprio=True,
    normalize_proprio=True,
    normalize_actions=True,
    num_duplicates_per_image=4,
    demonstration_sampling_prob=1.0,
    success_rollout_sampling_prob=0.0,
    return_value_function_returns=False,
    gamma=0.99,
)

cosmos_predict2_2b_480p_franka_shave_cucumber_20260321 = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_libero",
            "_self_",
        ],
        scheduler=dict(
            cycle_lengths=[20000, 100000000000000],
            warm_up_steps=[2000, 0],
            f_start=[1e-6, 0.06],
            f_max=[1.0, 0.06],
            f_min=[0.3, 0.06],
        ),
        model=L(CosmosPolicyVideo2WorldModel)(
            config=dict(
                state_t=12,
                min_num_conditional_frames=6,
                max_num_conditional_frames=6,
                tokenizer=dict(
                    chunk_duration=45,
                ),
                net=dict(
                    use_tactile_self_attn_bias=True,
                    tactile_self_attn_alpha=2.0,
                    tactile_latent_t_indices=(4, 5, 10, 11),
                    tactile_attn_chunk_q=32,
                ),
            ),
        ),
        dataloader_train=L(DataLoader)(
            num_workers=12,
            persistent_workers=True,
            pin_memory=True,
            dataset=franka_shave_cucumber_20260321_dataset,
            sampler=L(DistributedSampler)(
                dataset=franka_shave_cucumber_20260321_dataset,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=16,
            drop_last=True,
        ),
        checkpoint=dict(
            save_iter=50,
        ),
        job=dict(
            project="cosmos_policy_franka_shave_cucumber",
            group="cosmos_v2_finetune",
            name="cosmos_predict2_2b_480p_franka_shave_cucumber_20260321",
            wandb_id="franka_shave_cucumber_20260321",
        ),
    )
)

# Franka cut banana (tactile): same layout as shave_cucumber; T5 via save_franka_t5_text_embeddings on data_dir.
_CUT_BANANA_20260321_DIR = "/path/to/attention_data_cut_banana/cut_banana_20260321"
franka_cut_banana_20260321_dataset = L(FrankaDataset)(
    data_dir=_CUT_BANANA_20260321_DIR,
    t5_text_embeddings_path=os.path.join(_CUT_BANANA_20260321_DIR, "t5_embeddings.pkl"),
    chunk_size=20,
    use_image_aug=True,
    use_stronger_image_aug=True,
    use_wrist_images=True,
    use_third_person_images=True,
    use_proprio=True,
    normalize_proprio=True,
    normalize_actions=True,
    num_duplicates_per_image=4,
    demonstration_sampling_prob=1.0,
    success_rollout_sampling_prob=0.0,
    return_value_function_returns=False,
    gamma=0.99,
)

# Same HDF5 as tactile cut_banana, but no tactile load / 8 latent slots (LIBERO layout; baseline).
franka_cut_banana_20260321_no_tactile_dataset = L(FrankaDataset)(
    data_dir=_CUT_BANANA_20260321_DIR,
    t5_text_embeddings_path=os.path.join(_CUT_BANANA_20260321_DIR, "t5_embeddings.pkl"),
    chunk_size=20,
    use_image_aug=True,
    use_stronger_image_aug=True,
    use_wrist_images=True,
    use_third_person_images=True,
    use_proprio=True,
    normalize_proprio=True,
    normalize_actions=True,
    num_duplicates_per_image=4,
    demonstration_sampling_prob=1.0,
    success_rollout_sampling_prob=0.0,
    return_value_function_returns=False,
    gamma=0.99,
    use_tactile=False,
)

# Same as franka_cut_banana_20260321_dataset but no random aug on tactile (primary/wrist unchanged).
franka_cut_banana_20260321_no_tactile_img_aug_dataset = L(FrankaDataset)(
    data_dir=_CUT_BANANA_20260321_DIR,
    t5_text_embeddings_path=os.path.join(_CUT_BANANA_20260321_DIR, "t5_embeddings.pkl"),
    chunk_size=20,
    use_image_aug=True,
    use_stronger_image_aug=True,
    use_tactile_image_aug=False,
    use_wrist_images=True,
    use_third_person_images=True,
    use_proprio=True,
    normalize_proprio=True,
    normalize_actions=True,
    num_duplicates_per_image=4,
    demonstration_sampling_prob=1.0,
    success_rollout_sampling_prob=0.0,
    return_value_function_returns=False,
    gamma=0.99,
)

cosmos_predict2_2b_480p_franka_cut_banana_20260321 = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_libero",
            "_self_",
        ],
        scheduler=dict(
            cycle_lengths=[20000, 100000000000000],
            warm_up_steps=[2000, 0],
            f_start=[1e-6, 0.06],
            f_max=[1.0, 0.06],
            f_min=[0.3, 0.06],
        ),
        model=L(CosmosPolicyVideo2WorldModel)(
            config=dict(
                state_t=12,
                min_num_conditional_frames=6,
                max_num_conditional_frames=6,
                tokenizer=dict(
                    chunk_duration=45,
                ),
                net=dict(
                    use_tactile_self_attn_bias=True,
                    tactile_self_attn_alpha=2.0,
                    tactile_latent_t_indices=(4, 5, 10, 11),
                    tactile_attn_chunk_q=32,
                ),
            ),
        ),
        dataloader_train=L(DataLoader)(
            num_workers=12,
            persistent_workers=True,
            pin_memory=True,
            dataset=franka_cut_banana_20260321_dataset,
            sampler=L(DistributedSampler)(
                dataset=franka_cut_banana_20260321_dataset,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=16,
            drop_last=True,
        ),
        checkpoint=dict(
            save_iter=50,
        ),
        job=dict(
            project="cosmos_policy_franka_cut_banana",
            group="cosmos_v2_finetune",
            name="cosmos_predict2_2b_480p_franka_cut_banana_20260321",
            wandb_id="franka_cut_banana_20260321",
        ),
    )
)

# Same as cosmos_predict2_2b_480p_franka_cut_banana_20260321 but tactile streams skip image aug; new job path / W&B id.
cosmos_predict2_2b_480p_franka_cut_banana_20260321_no_tactile_img_aug = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_libero",
            "_self_",
        ],
        scheduler=dict(
            cycle_lengths=[20000, 100000000000000],
            warm_up_steps=[2000, 0],
            f_start=[1e-6, 0.06],
            f_max=[1.0, 0.06],
            f_min=[0.3, 0.06],
        ),
        model=L(CosmosPolicyVideo2WorldModel)(
            config=dict(
                state_t=12,
                min_num_conditional_frames=6,
                max_num_conditional_frames=6,
                tokenizer=dict(
                    chunk_duration=45,
                ),
                net=dict(
                    use_tactile_self_attn_bias=True,
                    tactile_self_attn_alpha=2.0,
                    tactile_latent_t_indices=(4, 5, 10, 11),
                    tactile_attn_chunk_q=32,
                ),
            ),
        ),
        dataloader_train=L(DataLoader)(
            num_workers=12,
            persistent_workers=True,
            pin_memory=True,
            dataset=franka_cut_banana_20260321_no_tactile_img_aug_dataset,
            sampler=L(DistributedSampler)(
                dataset=franka_cut_banana_20260321_no_tactile_img_aug_dataset,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=16,
            drop_last=True,
        ),
        checkpoint=dict(
            save_iter=50,
        ),
        job=dict(
            project="cosmos_policy_franka_cut_banana",
            group="cosmos_v2_finetune",
            name="cosmos_predict2_2b_480p_franka_cut_banana_20260321_no_tactile_img_aug",
            wandb_id="franka_cut_banana_20260321_no_tactile_img_aug",
        ),
    )
)

# Same 12-slot tactile inputs as above; standard self-attention (MiniTrainDIT use_tactile_self_attn_bias=False).
cosmos_predict2_2b_480p_franka_cut_banana_20260321_tactile_no_attn_bias = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_libero",
            "_self_",
        ],
        scheduler=dict(
            cycle_lengths=[20000, 100000000000000],
            warm_up_steps=[2000, 0],
            f_start=[1e-6, 0.06],
            f_max=[1.0, 0.06],
            f_min=[0.3, 0.06],
        ),
        model=L(CosmosPolicyVideo2WorldModel)(
            config=dict(
                state_t=12,
                min_num_conditional_frames=6,
                max_num_conditional_frames=6,
                tokenizer=dict(
                    chunk_duration=45,
                ),
                net=dict(
                    use_tactile_self_attn_bias=False,
                ),
            ),
        ),
        dataloader_train=L(DataLoader)(
            num_workers=12,
            persistent_workers=True,
            pin_memory=True,
            dataset=franka_cut_banana_20260321_dataset,
            sampler=L(DistributedSampler)(
                dataset=franka_cut_banana_20260321_dataset,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=16,
            drop_last=True,
        ),
        checkpoint=dict(
            save_iter=50,
        ),
        job=dict(
            project="cosmos_policy_franka_cut_banana",
            group="cosmos_v2_finetune",
            name="cosmos_predict2_2b_480p_franka_cut_banana_20260321_tactile_no_attn_bias",
            wandb_id="franka_cut_banana_20260321_tactile_no_attn_bias",
        ),
    )
)

cosmos_predict2_2b_480p_franka_cut_banana_20260321_no_tactile = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_libero",
            "_self_",
        ],
        scheduler=dict(
            cycle_lengths=[20000, 100000000000000],
            warm_up_steps=[2000, 0],
            f_start=[1e-6, 0.06],
            f_max=[1.0, 0.06],
            f_min=[0.3, 0.06],
        ),
        dataloader_train=L(DataLoader)(
            num_workers=12,
            persistent_workers=True,
            pin_memory=True,
            dataset=franka_cut_banana_20260321_no_tactile_dataset,
            sampler=L(DistributedSampler)(
                dataset=franka_cut_banana_20260321_no_tactile_dataset,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=16,
            drop_last=True,
        ),
        checkpoint=dict(
            save_iter=50,
        ),
        job=dict(
            project="cosmos_policy_franka_cut_banana",
            group="cosmos_v2_finetune",
            name="cosmos_predict2_2b_480p_franka_cut_banana_20260321_no_tactile",
            wandb_id="franka_cut_banana_20260321_no_tactile",
        ),
    )
)

# ALOHA planning model
# Dataset: 648 rollouts from evaluations with Cosmos Policy, pi05, pi0, OpenVLA-OFT+, Diffusion Policy
# NOTE: This rollouts dataset is not released; you will need to replace `rollout_data_dir` below with your own rollouts dataset
aloha_2025_09_18__648_rollouts__cosmos_policy__pi05__pi0__openvla_oft__diffusion_policy__dataset = L(
    ALOHADataset
)(
    data_dir=os.path.join(BASE_DATASETS_DIR, "ALOHA-Cosmos-Policy", "preprocessed"),
    t5_text_embeddings_path=os.path.join(BASE_DATASETS_DIR, "ALOHA-Cosmos-Policy", "preprocessed", "t5_embeddings.pkl"),
    chunk_size=50,
    use_image_aug=True,
    use_stronger_image_aug=True,
    use_proprio=True,
    normalize_proprio=True,
    normalize_actions=True,
    num_duplicates_per_image=4,  # WAN 2.1 tokenizer: 4 images per latent frame
    treat_demos_as_success_rollouts=False,  # Don't include demos as success rollouts because they have a fixed episode length + we want to focus on real policy rollouts
    demonstration_sampling_prob=0.1,  # Smaller demonstration sampling prob - more emphasis on rollouts
    success_rollout_sampling_prob=0.5,
    return_value_function_returns=True,
    gamma=0.998,  # Higher gamma for ALOHA because episodes can have up to 1.5-2.0K steps  # (s, a, s', v)
    rollout_data_dir=os.path.join(BASE_DATASETS_DIR, "PATH/TO/YOUR/ROLLOUTS/DATASET"),  # JPEG images
    use_jpeg_for_rollouts=True,  # JPEG images
)
cosmos_predict2_2b_480p_aloha_185_demos_4_tasks_mixture_foldshirt15_candiesinbowl45_candyinbag45_eggplantchickenonplate80__resumeFrom50K_648_rollouts_Vsprime_value_func = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_aloha_185_demos_4_tasks_mixture_foldshirt15_candiesinbowl45_candyinbag45_eggplantchickenonplate80",
            "_self_",
        ],
        checkpoint=dict(
            # Resume from 50K checkpoint of base Cosmos Policy run
            load_path=get_checkpoint_path(BASE_CHECKPOINT_PATH),
        ),
        scheduler=dict(
            # LR decay for 15K steps in cycle #1, then decay by 5x and stay constant forever in cycle #2
            cycle_lengths=[15000, 100000000000000],
            warm_up_steps=[1500, 0],
            f_start=[1e-6, 0.06],
            f_max=[1.0, 0.06],
            f_min=[0.3, 0.06],
        ),
        dataloader_train=L(DataLoader)(
            num_workers=12,
            persistent_workers=True,
            pin_memory=True,
            dataset=aloha_2025_09_18__648_rollouts__cosmos_policy__pi05__pi0__openvla_oft__diffusion_policy__dataset,
            sampler=L(DistributedSampler)(
                dataset=aloha_2025_09_18__648_rollouts__cosmos_policy__pi05__pi0__openvla_oft__diffusion_policy__dataset,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=25,
            drop_last=True,
        ),
        model=L(CosmosPolicyVideo2WorldModel)(
            config=dict(
                mask_current_state_action_for_value_prediction=True,  # Use input masking to mask out irrelevant inputs (current state and action) during value prediction
            ),
        ),
        job=dict(
            group="cosmos_v2_finetune",
            name="cosmos_predict2_2b_480p_aloha_185_demos_4_tasks_mixture_foldshirt15_candiesinbowl45_candyinbag45_eggplantchickenonplate80__resumeFrom50K_648_rollouts_Vsprime_value_func",
        ),
    )
)
# Inference version
cosmos_predict2_2b_480p_aloha_185_demos_4_tasks_mixture_foldshirt15_candiesinbowl45_candyinbag45_eggplantchickenonplate80__resumeFrom50K_648_rollouts_Vsprime_value_func__inference_only = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_aloha_185_demos_4_tasks_mixture_foldshirt15_candiesinbowl45_candyinbag45_eggplantchickenonplate80__resumeFrom50K_648_rollouts_Vsprime_value_func",
            "_self_",
        ],
        model=L(CosmosPolicyVideo2WorldModel)(
            config=dict(
                sde=L(HybridEDMSDE)(
                    sigma_max=80,
                    sigma_min=4,
                )
            )
        ),
        job=dict(
            group="cosmos_v2_inference",
            name="cosmos_predict2_2b_480p_aloha_185_demos_4_tasks_mixture_foldshirt15_candiesinbowl45_candyinbag45_eggplantchickenonplate80__resumeFrom50K_648_rollouts_Vsprime_value_func__inference_only",
        ),
    )
)


def register_configs():
    cs = ConfigStore.instance()
    # Register the experiments
    for _item in [
        # LIBERO
        cosmos_predict2_2b_480p_libero,  # *** Main checkpoint ***
        cosmos_predict2_2b_480p_libero__inference_only,
        # RoboCasa
        cosmos_predict2_2b_480p_robocasa_50_demos_per_task,  # *** Main checkpoint ***
        cosmos_predict2_2b_480p_robocasa_50_demos_per_task__inference,
        # ALOHA
        cosmos_predict2_2b_480p_aloha_185_demos_4_tasks_mixture_foldshirt15_candiesinbowl45_candyinbag45_eggplantchickenonplate80,  # *** Main checkpoint ***
        cosmos_predict2_2b_480p_aloha_185_demos_4_tasks_mixture_foldshirt15_candiesinbowl45_candyinbag45_eggplantchickenonplate80__inference_only,
        cosmos_predict2_2b_480p_franka_pick_and_place_baguette,  # Franka tactile data
        cosmos_predict2_2b_480p_franka_whiteboard,  # Franka whiteboard (state 6D, action 7D)
        cosmos_predict2_2b_480p_franka_hupai_tactile,  # Franka hupai + tactile (12 slots, state 6D, action 7D)
        cosmos_predict2_2b_480p_franka_shave_cucumber_20260321,  # Franka shave cucumber + tactile (same layout as hupai tactile)
        cosmos_predict2_2b_480p_franka_cut_banana_20260321,  # Franka cut banana + tactile (same layout as shave cucumber)
        cosmos_predict2_2b_480p_franka_cut_banana_20260321_no_tactile_img_aug,  # Cut banana + tactile; no aug on tactile RGB
        cosmos_predict2_2b_480p_franka_cut_banana_20260321_tactile_no_attn_bias,  # Tactile inputs, no tactile self-attn bias
        cosmos_predict2_2b_480p_franka_cut_banana_20260321_no_tactile,  # Same data dir, LIBERO 8-slot (no tactile load)
        cosmos_predict2_2b_480p_aloha_185_demos_4_tasks_mixture_foldshirt15_candiesinbowl45_candyinbag45_eggplantchickenonplate80__resumeFrom50K_648_rollouts_Vsprime_value_func,  # ALOHA planning model
        cosmos_predict2_2b_480p_aloha_185_demos_4_tasks_mixture_foldshirt15_candiesinbowl45_candyinbag45_eggplantchickenonplate80__resumeFrom50K_648_rollouts_Vsprime_value_func__inference_only,
    ]:
        experiment_name = _item["job"]["name"]
        log.info(f"Registering experiment: {experiment_name}")
        cs.store(
            group="experiment",
            package="_global_",
            name=experiment_name,
            node=_item,
        )
