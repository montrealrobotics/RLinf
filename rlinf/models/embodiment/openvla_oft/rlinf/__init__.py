# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os

import torch
from omegaconf import DictConfig
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoProcessor,
    AutoTokenizer,
)


def get_model_config_and_input_processor(cfg: DictConfig):
    from prismatic.extern.hf.configuration_prismatic import (
        OpenVLAConfig as OpenVLAOFTConfig,
    )

    from rlinf.models.embodiment.prismatic.processing_prismatic import (
        MultiInputPrismaticProcessor as PrismaticProcessorOFT,
    )
    from rlinf.models.embodiment.prismatic.processing_prismatic import (
        PrismaticImageProcessor,
    )

    AutoConfig.register("openvla", OpenVLAOFTConfig)
    AutoImageProcessor.register(OpenVLAOFTConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAOFTConfig, PrismaticProcessorOFT)

    model_config = OpenVLAOFTConfig.from_pretrained(
        cfg.model_path, center_crop=cfg.center_crop
    )
    image_processor = PrismaticImageProcessor.from_pretrained(
        cfg.model_path, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_path, trust_remote_code=True, padding_side="left"
    )
    input_processor = PrismaticProcessorOFT.from_pretrained(
        cfg.model_path,
        tokenizer=tokenizer,
        image_processor=image_processor,
        trust_remote_code=True,
    )
    return model_config, input_processor


def get_model(cfg: DictConfig, torch_dtype=torch.bfloat16):
    from prismatic.extern.hf.configuration_prismatic import (
        OpenVLAConfig as OpenVLAOFTConfig,
    )

    from rlinf.models.embodiment.openvla_oft.rlinf.openvla_oft_action_model import (
        OpenVLAOFTForRLActionPrediction,
    )

    AutoConfig.register("openvla", OpenVLAOFTConfig)
    actor_model_config = AutoConfig.from_pretrained(
        cfg.model_path, trust_remote_code=cfg.trust_remote_code
    )

    dataset_statistics_path = os.path.join(cfg.model_path, "dataset_statistics.json")
    if os.path.isfile(dataset_statistics_path):
        with open(dataset_statistics_path, "r") as f:
            new_norm_stats = json.load(f)
            norm_stats = getattr(actor_model_config, "norm_stats", {})
            norm_stats.update(new_norm_stats)
            setattr(actor_model_config, "norm_stats", norm_stats)

    override_config_kwargs = cfg
    if override_config_kwargs is not None:
        for key, val in override_config_kwargs.items():
            setattr(actor_model_config, key, val)

    # Load directly to the accelerator when the caller will place the model on
    # device.  This avoids a SIGSEGV on ROCm/AMD during the two-phase CPU→GPU
    # .to() call.  Use an explicit dict device_map (e.g. {"": "cuda:0"}) instead
    # of "auto" to skip accelerate's get_max_memory/get_balanced_memory, which
    # also segfaults on ROCm.  Worker.torch_device_type is "cuda" on both NVIDIA
    # and AMD (ROCm uses PyTorch's CUDA abstraction layer).
    # When load_to_device is False (e.g. FSDP actor expects a CPU model for
    # sharding), leave device_map as None so from_pretrained keeps weights on CPU.
    device_map = None
    if cfg.get("load_to_device", True):
        from rlinf.scheduler.worker.worker import Worker

        if Worker.torch_platform is not None and Worker.torch_platform.is_available():
            device_map = {"": f"{Worker.torch_device_type}:0"}

    model = OpenVLAOFTForRLActionPrediction.from_pretrained(
        pretrained_model_name_or_path=cfg.model_path,
        torch_dtype=torch_dtype,
        # attn_implementation="flash_attention_2",
        config=actor_model_config,
        action_dim=cfg.action_dim,
        num_action_chunks=cfg.num_action_chunks,
        trust_remote_code=True,
        add_value_head=cfg.add_value_head,
        max_prompt_length=cfg.max_prompt_length,
        device_map=device_map,
    )

    # oft add
    model.vision_backbone.set_num_images_in_input(cfg.get("num_images_in_input", 1))

    if device_map is None:
        model.to(torch_dtype)

    model_config, input_processor = get_model_config_and_input_processor(cfg)
    model.setup_config_and_processor(model_config, input_processor)

    return model
