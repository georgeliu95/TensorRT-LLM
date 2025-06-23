import torch
import numpy as np
import argparse
import time

from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
from diffusers.utils import export_to_video, load_image
from transformers import CLIPVisionModel

from tensorrt_llm._torch.models.modeling_wan import TllmWanTransformer3DModel


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="Wan-AI/Wan2.1-I2V-14B-480P-Diffusers")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    args = parser.parse_args()

    # Available models: Wan-AI/Wan2.1-I2V-14B-480P-Diffusers, Wan-AI/Wan2.1-I2V-14B-720P-Diffusers
    model_id = args.model_id
    image_encoder = CLIPVisionModel.from_pretrained(
        model_id, subfolder="image_encoder", torch_dtype=torch.float32
    )
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
    transformer = TllmWanTransformer3DModel.from_pretrained(model_id, subfolder="transformer", torch_dtype=torch.bfloat16)

    pipe = WanImageToVideoPipeline.from_pretrained(
        model_id, vae=vae, image_encoder=image_encoder, transformer=transformer, torch_dtype=torch.bfloat16
    )
    pipe.to("cuda")
    image = load_image(
        "https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P/resolve/main/examples/i2v_input.JPG"
    )
    max_area = 480 * 832
    aspect_ratio = image.height / image.width
    mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
    height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
    width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
    image = image.resize((width, height))
    prompt = (
        "Summer beach vacation style, a white cat wearing sunglasses \
        sits on a surfboard. The fluffy-furred feline gazes directly at the camera \
        with a relaxed expression. Blurred beach scenery forms the background featuring \
        crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. \
        The cat assumes a naturally relaxed posture, as if savoring the sea breeze and \
        warm sunlight. A close-up shot highlights the feline's intricate details and the \
        refreshing atmosphere of the seaside."
    )
    negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
    
    
    start_time = time.time()
    with torch.no_grad():
        output = pipe(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=81,
            guidance_scale=5.0,
            num_inference_steps=args.num_inference_steps,
            generator=torch.Generator(device="cuda").manual_seed(42),
            ).frames[0]
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    export_to_video(output, "diffusers.wan.output.mp4", fps=16)


if __name__ == "__main__":
    main()