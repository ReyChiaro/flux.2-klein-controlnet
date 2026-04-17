import io
import os
import sys
import copy
import torch
import argparse

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, DistributedDataParallelKwargs
from datetime import datetime
from loguru import logger

from torch.utils.data import Dataset, DataLoader

from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer

from diffusers.models.autoencoders.autoencoder_kl_flux2 import AutoencoderKLFlux2
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from diffusers.utils.torch_utils import is_compiled_module

from prodigyopt import Prodigy

from flux2.pipeline_flux2klein_controlnet import Flux2KleinControlNetPipeline
from flux2.transformer_flux2klein import Flux2Transformer2DModel
from flux2.controlnet_flux2klein import FLUX2KleinControlNetModel


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


def unwrap_model(accelerator: Accelerator, model: torch.nn.Module):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model


def parse_args() -> tuple[argparse.Namespace, str]:
    parser = argparse.ArgumentParser("ControlNet of FLUX.2-Klein-9B Training")
    timestamp = datetime.time().strftime("%Y%m%d-%H%M%S")
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

    # ---------------- Train ---------------- #
    parser.add_argument("--seed", type=int, default=42, help="Choose a number you like, I like 42.")
    parser.add_argument("--max-training-steps", type=int, help="Max training steps for controlnet.")
    parser.add_argument("--mixed-precision", type=str, choices=["no", "bf16", "fp16"], default="bf16")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--log-with", type=str, choices=["tensorboard", "wandb"], default="tensorboard")
    parser.add_argument("--gradient-checkpointing", action="store_true", default=False)

    return parser.parse_args(), timestamp


def train(args: argparse.Namespace, timestamp: str):
    output_dir = os.path.join(args.output_dir, args.project_name, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    checkpoint_dir = os.path.join(output_dir, args.checkpoint_dir)
    evaluation_dir = os.path.join(output_dir, args.evaluation_dir)
    log_dir = os.path.join(output_dir, args.log_dir)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with=args.low_with,
        project_config=ProjectConfiguration(project_dir=output_dir, logging_dir=log_dir),
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )

    RANK = accelerator.process_index
    WORLD_SIZE = accelerator.num_processes
    DEVICE = torch.device(f"cuda:{RANK}")
    IS_MAIN_PROCESS = accelerator.is_main_process
    WEIGHT_DTYPE = torch.float32
    if accelerator.mixed_precision == "bf16":
        WEIGHT_DTYPE = torch.bfloat16
    if accelerator.mixed_precision == "f1p6":
        WEIGHT_DTYPE = torch.float16

    if IS_MAIN_PROCESS:
        accelerator.init_trackers(project_name=args.project_name)

    logger = setup_logger(IS_MAIN_PROCESS, RANK, log_dir, log_filename=args.project_name, log_per_rank=True)

    # TODO: dataset configs
    trainset = Dataset()
    evalset = Dataset()
    # TODO: We suppose the data can be batched with same shapes
    train_loader = DataLoader(trainset)

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
        text_encoder = Qwen3ForCausalLM.from_pretrained(
            pretrained_model_name_or_path=args.base_model,
            subfolder="text_encoder",
            torch_dtype=WEIGHT_DTYPE,
            device_map=DEVICE,
        )
        text_encoder.requires_grad_(False)
    vae = AutoencoderKLFlux2.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
    )
    transformer = Flux2Transformer2DModel.from_pretrained(
        pretrained_model_name_or_path=args.base_model,
        subfolder="transformer",
        torch_dtype=WEIGHT_DTYPE,
        device_map=DEVICE,
    )
    if args.controlnet:
        controlnet = FLUX2KleinControlNetModel.from_pretrained(
            pretrained_model_name_or_path=args.controlnet,
            torch_dtype=WEIGHT_DTYPE,
            device_map=DEVICE,
        )
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

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
        controlnet.enable_gradient_checkpointing()
    
    # ---------------- Optimizer ---------------- #
    optimizer = Prodigy(controlnet.parameters())

if __name__ == "__main__":
    train(*parse_args())
