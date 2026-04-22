import os
import sys
import copy
import json
import math
import torch
import random
import argparse
import numpy as np
import torchvision.transforms as T

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, DistributedDataParallelKwargs
from datetime import datetime
from loguru import logger
from PIL import Image
from safetensors.torch import load_file, save_file

from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision.utils import save_image

from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer

from diffusers.models.autoencoders.autoencoder_kl_flux2 import AutoencoderKLFlux2
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.training_utils import compute_loss_weighting_for_sd3, compute_density_for_timestep_sampling, free_memory

from prodigyopt import Prodigy

from flux2.pipeline_flux2klein_controlnet import Flux2KleinControlNetPipeline
from flux2.transformer_flux2klein import Flux2Transformer2DModel
from flux2.controlnet_flux2klein import FLUX2KleinControlNetModel


class ControlNetImageEditDataset(Dataset):
    r"""
    Dataset used for controlnet with conditioning inputs.

    The dataset *must* have following keys:
    - source_images: The images that ready to be edited,
    - control_images: The images that will be applied to, as the inputs of controlnet,
    - target_images: The target edited images,
    - prompts: The prompts to pipeline/text encoder.

    The following keys are *optional*:
    - prompt_embeds: The encoded prompts by text encoder.
    """

    def __init__(self, data_file: str, base_resolution: int, bucket_data: bool = False, data_root: str | None = None):
        r"""
        Args:
            data_file: Path to the data mapping information, JSON format.
                1. If not bucket dataset, the item of dataset should be list of
                {
                    "source_images": <path to image>,
                    "control_images": <path to image>,
                    "target_images": <path to image>,
                    "prompts": str,
                }
                In this case, the batch size is fixed to 1.
                2. If dataset has been mapped to bucket to enable batch size training, the
                item of dataset should be dict of:
                {
                    <bucket_id>: {
                        "aspect_ratio": tuple[int, int],
                        "dataset": [
                            {
                                "source_images": <path to image>,
                                "control_images": <path to image>,
                                "target_images": <path to image>,
                                "prompts": str,
                            }
                        ] // List of data item, each item is same of normal data
                    } // Each bucket should be assigned an aspect ratio
                }
        """
        super().__init__()
        with open(data_file, "r") as f:
            self.raw_data = json.load(f)

        self.data_root = data_root
        self.bucket_data = bucket_data
        self.base_resolution = base_resolution
        self.items = []
        self.item_to_bucket = []

        if not self.bucket_data:
            self.items = self.raw_data
        else:
            for bucket_id, content in self.raw_data.items():
                # aspect_ratio is now [W_ratio, H_ratio] e.g., [16, 9]
                ratio = content["aspect_ratio"]
                if ratio is None:
                    continue
                target_size = self._calculate_target_size(ratio)

                for entry in content["dataset"]:
                    entry["target_size"] = target_size
                    self.items.append(entry)
                    self.item_to_bucket.append(bucket_id)

        self.base_transform = T.Compose([T.ToTensor()])

    def construct_prompt(
        self,
        prompt: str,
        placeholder: str | None = "[TARGET]",
        replaced_words: str | None = None,
    ) -> str:
        return prompt if placeholder is None or replaced_words is None else prompt.replace(placeholder, replaced_words)

    def _calculate_target_size(self, ratio: list[int]) -> tuple[int, int]:
        r"""
        Converts a ratio like [16, 9] into pixel dimensions.
        Calculates dimensions such that the total area is roughly (base_res^2),
        keeping both sides divisible by 8 or 64 for transformer/VAE compatibility.
        """
        rw, rh = ratio
        # Scale factor to maintain approximate pixel area of base_res * base_res
        scale = (self.base_resolution / (rw * rh)) ** 0.5
        target_w = int(round(rw * scale / 64.0)) * 64
        target_h = int(round(rh * scale / 64.0)) * 64
        return (max(64, target_w), max(64, target_h))

    def __len__(self):
        return len(self.items)

    def _crop_to_aspect_and_resize(self, image: Image.Image, target_size: tuple[int, int]) -> Image.Image:
        r"""
        Crops the image to the target aspect ratio from the center, then resizes.
        """
        nw, nh = target_size
        target_aspect = nw / nh

        width, height = image.size
        current_aspect = width / height

        if current_aspect > target_aspect:
            # Image is too wide: reduce width
            new_width = int(target_aspect * height)
            left = (width - new_width) // 2
            top = 0
            right = left + new_width
            bottom = height
        else:
            # Image is too tall: reduce height
            new_height = int(width / target_aspect)
            left = 0
            top = (height - new_height) // 2
            right = width
            bottom = top + new_height

        # Crop and then resize to the bucket resolution
        image = image.crop((left, top, right, bottom))
        return image.resize((nw, nh), Image.LANCZOS)

    def __getitem__(self, index):
        item = self.items[index]

        # Load images
        source_path = (
            os.path.join(self.data_root, item["source_images"]) if self.data_root is not None else item["source_images"]
        )
        control_path = (
            os.path.join(self.data_root, item["control_images"])
            if self.data_root is not None
            else item["control_images"]
        )
        target_path = (
            os.path.join(self.data_root, item["target_images"]) if self.data_root is not None else item["target_images"]
        )
        source_img = Image.open(source_path).convert("RGB")
        control_img = Image.open(control_path).convert("RGB")
        target_img = Image.open(target_path).convert("RGB")

        if "target_size" in item:
            tw, th = item["target_size"]
        else:
            tw, th = target_img.size

        # 1. Target is likely already the correct aspect, but we resize it to be sure
        # (Note: If target is already cropped, direct resize is fine)
        target_img = target_img.resize((tw, th), Image.LANCZOS)

        # 2. Source and Control (Mask) are center-cropped to target's aspect, then resized
        source_img = self._crop_to_aspect_and_resize(source_img, (tw, th))
        control_img = self._crop_to_aspect_and_resize(control_img, (tw, th))

        source_img = self.base_transform(source_img)
        control_img = self.base_transform(control_img)
        target_img = self.base_transform(target_img)

        # Optional: Convert to Tensors here if not using a separate transform
        return {
            "source_images": source_img,
            "control_images": control_img,
            "target_images": target_img,
            "prompts": self.construct_prompt(item["prompts"], replaced_words="the masked object"),
        }


class BucketBatchSampler(Sampler):
    r"""
    An accelerate compatible batch sampler, support DDP training.
    Note that we should initialize accelerator with `dispatch_batches=False`.
    """

    def __init__(
        self,
        item_to_bucket: list[int],
        batch_size: int,
        num_replicas: int = 1,
        rank: int = 0,
        drop_last: bool = False,
        seed: int = 42,
    ):
        r"""
        :param num_replicas (int, default 1): Set to `num_processes` or `ngpu`. Used to split
            batches into different processes in DDP training.
        """
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.drop_last = drop_last

        self.bucket_indices = {}
        for idx, bucket_id in enumerate(item_to_bucket):
            if bucket_id not in self.bucket_indices:
                self.bucket_indices[bucket_id] = []
            self.bucket_indices[bucket_id].append(idx)

        self.batches = self._prepare_batches()
        logger.info(f"BatchSampler: Prepared {len(self.batches)} batches.")

    def _prepare_batches(self):
        batches = []
        for _, sids in self.bucket_indices.items():
            random.seed(self.seed)
            random.shuffle(sids)

            for i in range(0, len(sids), self.batch_size):
                batch = sids[i : i + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)

        random.seed(self.seed)
        random.shuffle(batches)

        return batches[self.rank :: self.num_replicas]

    def __iter__(self):
        for batch in self.batches:
            yield batch

    def __len__(self) -> int:
        return len(self.batches)


def setup_logger(is_main_process: bool, rank: int, log_dir: str, log_filename: str, log_per_rank: bool):
    logger.remove()
    log_path = os.path.join(log_dir, f"{log_filename}-rank{rank}.log")

    if is_main_process:
        logger.add(
            sys.stderr,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level="INFO",
            colorize=True,
        )
        logger.add(
            log_path,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} - {message}",
            level="DEBUG",
            rotation="50 MB",
            enqueue=True,
        )
    elif log_per_rank:
        logger.add(
            log_path,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} - {message}",
            level="DEBUG",
            rotation="50 MB",
            enqueue=True,
        )

    return logger


def parse_args() -> tuple[argparse.Namespace, str]:
    parser = argparse.ArgumentParser("ControlNet of FLUX.2-Klein-9B Training")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    logger.info(f"Timestamp: {timestamp}")

    # ---------------- Folder --------------- #
    parser.add_argument("--project-name", type=str, default="train-flux2-klein-controlnet")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Project output dir.")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Created under outputs dir.")
    parser.add_argument("--evaluation-dir", type=str, default="evaluation", help="Created under outputs dir.")
    parser.add_argument("--log-dir", type=str, default="logs", help="Created under outputs dir.")

    # ---------------- Model ---------------- #
    parser.add_argument("--base-model", type=str, help="Base pretrained model name or path.")
    parser.add_argument("--load-text-encoder", action="store_true", default=False)
    parser.add_argument("--controlnet", type=str, help="Pretrained controlnet for base model.")
    parser.add_argument("--num-controlnet-layers", type=int, default=4)
    parser.add_argument("--num-controlnet-single-layers", type=int, default=10)
    parser.add_argument("--conditioning-scale", type=float, default=1.0)

    # ---------------- Data ----------------- #
    parser.add_argument("--data-file", type=str, help="Path to data file, JSON format.")
    parser.add_argument("--base-resolution", type=int, default=1024 * 1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--bucket-data", action="store_true", default=False, help="Whether use bucket dataset.")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--data-root", type=str, default=None)

    # ---------------- Train ---------------- #
    parser.add_argument("--seed", type=int, default=42, help="Choose a number you like, I like 42.")
    parser.add_argument("--max-training-steps", type=int, help="Max training steps for controlnet.")
    parser.add_argument("--mixed-precision", type=str, choices=["no", "bf16", "fp16"], default="bf16")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--log-with", type=str, choices=["tensorboard", "wandb"], default="tensorboard")
    parser.add_argument("--gradient-checkpointing", action="store_true", default=False)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--weighting-scheme", type=str, default="logit_normal")
    parser.add_argument("--logit-mean", type=float, default=0.0)
    parser.add_argument("--logit-std", type=float, default=1.0)
    parser.add_argument("--mode-scale", type=float, default=1.29)
    parser.add_argument("--save-steps", type=int, help="Num of gradient update steps to save model.")
    parser.add_argument("--eval-steps", type=int, help="Num of gradient update steps to eval model.")
    parser.add_argument("--num-eval", type=int, default=5)

    return parser.parse_args(), timestamp


def unwrap_model(accelerator: Accelerator, model: torch.nn.Module):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model


def get_sigmas(
    scheduler: FlowMatchEulerDiscreteScheduler,
    timesteps: torch.Tensor,
    ndim: int,
    device: torch.device,
    weight_dtype: torch.dtype,
) -> torch.Tensor:
    sigmas: torch.Tensor = scheduler.sigmas.to(device=device, dtype=weight_dtype)
    all_timesteps: torch.Tensor = scheduler.timesteps.to(device=device, dtype=weight_dtype)

    timesteps = timesteps.to(device=device, dtype=weight_dtype)
    indices = [(all_timesteps == t).nonzero().item() for t in timesteps]

    sigmas = sigmas[indices].flatten()
    while sigmas.ndim < ndim:
        sigmas = sigmas.unsqueeze(-1)
    return sigmas


def encode_prompt(
    pipeline: Flux2KleinControlNetPipeline,
    prompt: list[str] | str,
    device: torch.device,
    weight_dtype: torch.dtype,
    prompt_embeds: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    prompt_embeds, text_ids = pipeline.encode_prompt(
        prompt=prompt,
        device=device,
        prompt_embeds=prompt_embeds,
    )
    prompt_embeds = prompt_embeds.to(device, weight_dtype)
    text_ids = text_ids.to(device, weight_dtype)
    return prompt_embeds, text_ids


def log_validation(
    args,
    accelerator: Accelerator,
    transformer: Flux2Transformer2DModel,
    controlnet: FLUX2KleinControlNetModel,
    eval_samples: list[dict[str, torch.Tensor | str]],
    global_step: int,
    evaluation_dir: str,
    device: torch.device,
    weight_dtype: torch.dtype,
    is_final_validation=False,
    final_checkpoint: str | None = None,
):
    # Fix bfloat16
    weight_dtype = torch.bfloat16
    evaluation_dir = os.path.join(evaluation_dir, f"step-{global_step}")
    os.makedirs(evaluation_dir, exist_ok=True)

    if not is_final_validation or final_checkpoint is None:
        controlnet = accelerator.unwrap_model(controlnet)
        pipeline = Flux2KleinControlNetPipeline.from_pretrained(
            pretrained_model_name_or_path=args.base_model,
            controlnet=controlnet,
            transformer=transformer,
            torch_dtype=weight_dtype,
        )
    else:
        controlnet = FLUX2KleinControlNetModel.from_pretrained(
            args.output_dir,
            torch_dtype=weight_dtype,
            device_map=device,
        )
        pipeline = Flux2KleinControlNetPipeline.from_pretrained(
            pretrained_model_name_or_path=args.base_model,
            controlnet=controlnet,
            transformer=transformer,
            torch_dtype=weight_dtype,
        )

    pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    image_logs: list[dict[str, str | Image.Image]] = []
    autocast_ctx = torch.autocast(accelerator.device.type)

    for i, sample in enumerate(eval_samples):
        source_image = sample.get("source_images", None)
        control_image = sample.get("control_images", None)
        target_image = sample.get("target_images", None)
        prompt = sample.get("prompts", None)

        source_image = T.ToPILImage()(source_image)
        control_image = T.ToPILImage()(control_image)
        target_image = T.ToPILImage()(target_image)

        with autocast_ctx:
            image = pipeline(
                image=source_image,
                mask=control_image,
                conditioning_scale=args.conditioning_scale,
                prompt=prompt,
                num_inference_steps=50,
                height=target_image.height,
                width=target_image.width,
                generator=generator,
                output_type="pil",
                return_dict=True,
            ).images[0]

        image_logs.append(
            {
                "source_image": source_image,
                "control_image": control_image,
                "target_image": target_image,
                "predict_image": image,
                "prompt": prompt,
            }
        )

        source_image = T.ToTensor()(source_image)
        control_image = T.ToTensor()(control_image)
        target_image = T.ToTensor()(target_image)
        image = T.ToTensor()(image)
        save_image(
            torch.cat([source_image, control_image, target_image, image], dim=-1),
            os.path.join(evaluation_dir, f"eval-{i}.jpg"),
        )

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            for log in image_logs:
                prompt = log["prompt"]
                log.pop("prompt")
                formatted_images = np.stack(list(log.values()))
                tracker.writer.add_images(prompt, formatted_images, global_step, dataformats="NHWC")

    del pipeline
    free_memory()
    return image_logs


def train(args: argparse.Namespace, timestamp: str):
    # ---------------- Initialize ---------------- #
    output_dir = os.path.join(args.output_dir, args.project_name, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    checkpoint_dir = os.path.join(output_dir, args.checkpoint_dir)
    evaluation_dir = os.path.join(output_dir, args.evaluation_dir)
    log_dir = os.path.join(output_dir, args.log_dir)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with=args.log_with,
        project_config=ProjectConfiguration(project_dir=output_dir, logging_dir=log_dir),
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )

    RANK = accelerator.process_index
    WORLD_SIZE = accelerator.num_processes
    DEVICE = torch.device(f"cuda:{RANK}")
    GENERATOR = torch.Generator(DEVICE).manual_seed(args.seed)
    IS_MAIN_PROCESS = accelerator.is_main_process
    WEIGHT_DTYPE = torch.float32
    if accelerator.mixed_precision == "bf16":
        WEIGHT_DTYPE = torch.bfloat16
    if accelerator.mixed_precision == "fp16":
        WEIGHT_DTYPE = torch.float16

    if IS_MAIN_PROCESS:
        accelerator.init_trackers(project_name=args.project_name, config=dict(vars(args)))

    logger = setup_logger(IS_MAIN_PROCESS, RANK, log_dir, log_filename=args.project_name, log_per_rank=True)

    # ---------------- Data -------------------- #
    trainset = ControlNetImageEditDataset(
        data_file=args.data_file,
        base_resolution=args.base_resolution,
        bucket_data=args.bucket_data,
        data_root=args.data_root,
    )
    evalset = copy.deepcopy(trainset)
    train_loader = DataLoader(
        trainset,
        batch_sampler=BucketBatchSampler(
            trainset.item_to_bucket,
            batch_size=args.batch_size,
            rank=RANK,
            num_replicas=WORLD_SIZE,
            seed=args.seed,
        ),
        num_workers=args.num_workers,
    )
    # train_loader = DataLoader(
    #     trainset,
    #     batch_size=args.batch_size,
    #     shuffle=True,
    #     num_workers=args.num_workers,
    #     drop_last=False,
    # )

    # ---------------- Pipeline ---------------- #
    tokenizer = None
    text_encoder = None
    if args.load_text_encoder:
        tokenizer = Qwen2Tokenizer.from_pretrained(
            pretrained_model_name_or_path=args.base_model,
            subfolder="tokenizer",
            torch_dtype=WEIGHT_DTYPE,
            device_map=DEVICE,
        )
        logger.info(f"Load tokenizer to {WEIGHT_DTYPE} on {DEVICE}")
        text_encoder = Qwen3ForCausalLM.from_pretrained(
            pretrained_model_name_or_path=args.base_model,
            subfolder="text_encoder",
            torch_dtype=WEIGHT_DTYPE,
            device_map=DEVICE,
        )
        text_encoder.requires_grad_(False)
        logger.info(f"Load text_encoder to {WEIGHT_DTYPE} on {DEVICE}")
    vae = AutoencoderKLFlux2.from_pretrained(
        pretrained_model_name_or_path=args.base_model,
        subfolder="vae",
        torch_dtype=WEIGHT_DTYPE,
        device_map=DEVICE,
    )
    logger.info(f"Load vae to {WEIGHT_DTYPE} on {DEVICE}")
    transformer = Flux2Transformer2DModel.from_pretrained(
        pretrained_model_name_or_path=args.base_model,
        subfolder="transformer",
        torch_dtype=WEIGHT_DTYPE,
        device_map=DEVICE,
    )
    logger.info(f"Load transformer to {WEIGHT_DTYPE} on {DEVICE}")
    if args.controlnet:
        controlnet = FLUX2KleinControlNetModel.from_pretrained(
            pretrained_model_name_or_path=args.controlnet,
            torch_dtype=WEIGHT_DTYPE,
            device_map=DEVICE,
        )
        logger.info(f"Load controlnet from {args.controlnet} to {WEIGHT_DTYPE} on {DEVICE}")
    else:
        controlnet = FLUX2KleinControlNetModel.from_transformer(
            transformer,
            attention_head_dim=transformer.config["attention_head_dim"],
            num_attention_heads=transformer.config["num_attention_heads"],
            num_layers=args.num_controlnet_layers,
            num_single_layers=args.num_controlnet_single_layers,
            load_weights_from_transformer=True,
        )
        controlnet.to(DEVICE, WEIGHT_DTYPE)
        logger.info(f"Load controlnet from transformer to {WEIGHT_DTYPE} on {DEVICE}")

    # Copy scheduler to devide train and eval phase, avoiding shared timesteps.
    eval_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        pretrained_model_name_or_path=args.base_model,
        subfolder="scheduler",
    )
    train_scheduler = copy.deepcopy(eval_scheduler)
    vae.requires_grad_(False)
    transformer.requires_grad_(False)
    controlnet.train()

    pipeline = Flux2KleinControlNetPipeline(
        scheduler=eval_scheduler,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        transformer=transformer,
        controlnet=controlnet,
    ).to(DEVICE)
    logger.info(f"Pipelin initialized.")

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
        controlnet.enable_gradient_checkpointing()

    # ---------------- Optimizer ---------------- #
    optimizer = Prodigy(controlnet.parameters())
    logger.info(f"Optimizer: Prodigy")

    # ---------------- Train Args ---------------- #
    controlnet, optimizer, train_loader = accelerator.prepare(controlnet, optimizer, train_loader)

    update_steps_per_epoch = math.ceil(len(train_loader) / accelerator.gradient_accumulation_steps)
    num_epochs = math.ceil(args.max_training_steps / update_steps_per_epoch)
    total_batch_size = args.batch_size * WORLD_SIZE * accelerator.gradient_accumulation_steps

    log_title = "=" * 25 + " Start Task " + "=" * 25
    log_content = f"\n{log_title}"
    log_content += f"\n  World size: {WORLD_SIZE}"
    log_content += f"\n  Mixed precision: {accelerator.mixed_precision}"
    log_content += f"\n  Num samples: {len(trainset)}"
    log_content += f"\n  Batch size per rank: {args.batch_size}"
    log_content += f"\n  Total batch size: {total_batch_size}"
    log_content += f"\n  Num epochs: {num_epochs}"
    log_content += f"\n  Num update steps per epoch: {update_steps_per_epoch}"
    log_content += f"\n  Num update steps: {args.max_training_steps}"
    log_content += f"\n  Gradient accumulation steps: {accelerator.gradient_accumulation_steps}"
    log_content += "\n" + "=" * len(log_title)
    logger.info(log_content)

    global_steps = 0
    align_bit = len(str(args.max_training_steps))

    # TODO: Add resume from checkpoint if required
    # ...

    num_latent_channels = pipeline.transformer.config.in_channels // 4
    for epoch in range(num_epochs):
        for batch_idx, batch in enumerate(train_loader):
            with accelerator.accumulate(controlnet):
                prompts: list[str] = batch.get("prompts", "")
                source_images: torch.Tensor = batch.get("source_images", None)
                control_images: torch.Tensor = batch.get("control_images", None)
                target_images: torch.Tensor = batch.get("target_images", None)
                prompt_embeds: torch.Tensor | None = batch.get("prompt_embeds", None)

                prompt_embeds, text_ids = pipeline.encode_prompt(
                    prompt=prompts,
                    device=DEVICE,
                    num_images_per_prompt=1,
                    prompt_embeds=prompt_embeds,
                )
                prompt_embeds = prompt_embeds.to(DEVICE, WEIGHT_DTYPE)
                # TODO: Add CFG if required
                # ...

                # source_images = pipeline.prepare_images(source_images, pipeline.image_processor)
                # control_images = pipeline.prepare_images(control_images, pipeline.mask_processor)

                B, _, H, W = target_images.shape

                noise_latents, noise_ids = pipeline.prepare_latents(
                    batch_size=B,
                    num_latents_channels=num_latent_channels,
                    height=H,
                    width=W,
                    dtype=WEIGHT_DTYPE,
                    device=DEVICE,
                    generator=GENERATOR,
                )
                source_latents, source_ids = pipeline.prepare_image_latents(
                    images=[source_images],
                    batch_size=B,
                    generator=GENERATOR,
                    device=DEVICE,
                    dtype=WEIGHT_DTYPE,
                )
                target_latents, target_ids = pipeline.prepare_image_latents(
                    images=[target_images],
                    batch_size=B,
                    generator=GENERATOR,
                    device=DEVICE,
                    dtype=WEIGHT_DTYPE,
                )
                control_latents, control_latent_ids = pipeline.prepare_control_latents(
                    cond_images=[source_images],
                    mask_images=[control_images],
                    batch_size=B,
                    generator=GENERATOR,
                    device=DEVICE,
                    dtype=WEIGHT_DTYPE,
                )

                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=B,
                    logit_mean=args.logit_mean,
                    logit_std=args.logit_std,
                    mode_scale=args.mode_scale,
                    device=DEVICE,
                    generator=GENERATOR,
                )
                indices = (u * train_scheduler.config.num_train_timesteps).long().cpu()
                timesteps = train_scheduler.timesteps.cpu()
                timesteps = timesteps[indices].to(DEVICE)
                sigmas = get_sigmas(
                    train_scheduler,
                    timesteps,
                    target_latents.ndim,
                    device=DEVICE,
                    weight_dtype=WEIGHT_DTYPE,
                )

                noisy_latents = (1.0 - sigmas) * target_latents + sigmas * noise_latents
                pred_targets = noise_latents - target_latents

                hidden_states = torch.cat([noisy_latents, source_latents], dim=1)
                latent_image_ids = torch.cat([target_ids, source_ids], dim=1)

                with accelerator.autocast():
                    # ControlNet
                    controlnet_block_samples, controlnet_single_block_samples = controlnet(
                        hidden_states=hidden_states,
                        controlnet_cond=control_latents,
                        timestep=timesteps / 1000,
                        conditioning_scale=args.conditioning_scale,
                        encoder_hidden_states=prompt_embeds,
                        img_ids=latent_image_ids,
                        txt_ids=text_ids,
                        guidance=None,
                        return_dict=False,
                    )

                    # Transformer, one step denoise
                    pred_vs = transformer(
                        hidden_states=hidden_states,
                        timestep=timesteps / 1000,
                        encoder_hidden_states=prompt_embeds,
                        img_ids=latent_image_ids,
                        txt_ids=text_ids,
                        guidance=None,
                        controlnet_block_samples=controlnet_block_samples,
                        controlnet_single_block_samples=controlnet_single_block_samples,
                        return_dict=False,
                    )[0]
                pred_vs = pred_vs[:, : noisy_latents.shape[1], ...]

                loss_weights = compute_loss_weighting_for_sd3(
                    weighting_scheme=args.weighting_scheme,
                    sigmas=sigmas,
                )
                loss = torch.mean(
                    (loss_weights.float() * (pred_vs.float() - pred_targets.float()) ** 2).reshape(B, -1),
                    dim=1,
                )
                loss = loss.mean()
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    params_to_clip = controlnet.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                optimizer.step()
                optimizer.zero_grad()
            # End accumulation

            if accelerator.sync_gradients:
                global_steps += 1
                # DeepSpeed requires saving weights on every device; saving weights only on the main process would cause issues.
                if IS_MAIN_PROCESS and global_steps % args.save_steps == 0:
                    save_path = os.path.join(
                        checkpoint_dir,
                        f"checkpoint-{global_steps:0{align_bit}d}.safetensors",
                    )
                    unwrap_controlnet = unwrap_model(accelerator, controlnet).to(torch.float32)
                    state_dicts = unwrap_controlnet.state_dict()
                    save_file(state_dicts, save_path)
                    logger.info(f"Save state to {save_path}")

                if IS_MAIN_PROCESS and global_steps % args.eval_steps == 0:
                    random_indices = random.choices(list(range(len(evalset))), k=args.num_eval)
                    eval_samples = [evalset[i] for i in random_indices]
                    image_logs = log_validation(
                        args=args,
                        accelerator=accelerator,
                        transformer=transformer,
                        controlnet=controlnet,
                        eval_samples=eval_samples,
                        global_step=global_steps,
                        evaluation_dir=evaluation_dir,
                        device=DEVICE,
                        weight_dtype=WEIGHT_DTYPE,
                        is_final_validation=False,
                        final_checkpoint=None,
                    )
            logs = {"loss": loss.detach().item()}
            logger.info(f"Step [{global_steps:0{align_bit}d}/{args.max_training_steps}] | Loss {loss:.6f}")
            accelerator.log(logs, step=global_steps)

            if global_steps >= args.max_training_steps:
                break
        # End batch
    # End epoch
    accelerator.wait_for_everyone()

    if IS_MAIN_PROCESS:
        save_path = os.path.join(checkpoint_dir, "lastest")
        os.makedirs(save_path, exist_ok=True)
        unwrap_controlnet: FLUX2KleinControlNetModel = unwrap_model(accelerator, controlnet).to(torch.float32)
        unwrap_controlnet.save_pretrained(save_path)

        random_indices = random.choices(list(range(len(evalset))), k=args.num_eval)
        eval_samples = [evalset[i] for i in random_indices]
        image_logs = log_validation(
            args=args,
            accelerator=accelerator,
            transformer=transformer,
            controlnet=controlnet,
            eval_samples=eval_samples,
            global_step=global_steps,
            evaluation_dir=evaluation_dir,
            device=DEVICE,
            weight_dtype=WEIGHT_DTYPE,
            is_final_validation=True,
            final_checkpoint=save_path,
        )
    accelerator.end_training()


if __name__ == "__main__":
    train(*parse_args())
