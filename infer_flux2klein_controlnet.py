import torch
import argparse
import torchvision.transforms.functional as T

from PIL import Image

from flux2.pipeline_flux2klein_controlnet import Flux2KleinControlNetPipeline
from flux2.controlnet_flux2klein import FLUX2KleinControlNetModel
from flux2.transformer_flux2klein import Flux2Transformer2DModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Flux2KleinControlNetPipeline ArgumentParser")
    parser.add_argument("--model", type=str, default="black-forest-labs/FLUX.2-Klein-9B")
    parser.add_argument("--controlnet", type=str)
    parser.add_argument("--conditioning-scale", type=float, default=1.0)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = "cuda:7"
    weight_dtype = torch.bfloat16

    transformer = Flux2Transformer2DModel.from_pretrained(
        pretrained_model_name_or_path=args.model,
        subfolder="transformer",
        torch_dtype=weight_dtype,
    ).to(device)
    controlnet = FLUX2KleinControlNetModel.from_transformer(transformer)
    controlnet.to(device=device, dtype=weight_dtype)

    pipeline = Flux2KleinControlNetPipeline.from_pretrained(
        pretrained_model_name_or_path=args.model,
        transformer=transformer,
        controlnet=controlnet,
        torch_dtype=weight_dtype,
    ).to(device)

    # image = T.to_pil_image(torch.randn((3, 1024, 1024)))
    # mask = T.to_pil_image(torch.randint(low=0, high=2, size=(3, 1024, 1024), dtype=torch.float))
    image = Image.open("assets/source_000000001360.jpg")
    mask = Image.open("assets/mask_000000001360_0.png")
    output = pipeline(
        image=image,
        mask=mask,
        conditioning_scale=args.conditioning_scale,
        prompt="the dog is wearing a red scarf.",
        num_inference_steps=50,
        generator=torch.Generator(device).manual_seed(42),
        output_type="pil",
        return_dict=True,
    ).images[0]

    output.save("output.jpg")
