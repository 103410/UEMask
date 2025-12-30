
import argparse
import json
import logging
import os
import random
import time
from pathlib import Path
from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from PIL import Image
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
)
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

# Set matplotlib backend to prevent errors in headless environments
matplotlib.use('Agg')

logger = get_logger(__name__)


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    """
    Helper function to import the correct text encoder class based on the model name.
    """
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel
        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation
        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")


class CustomDiffusionDataset(Dataset):
    """
    A dataset to prepare the instance images with the prompts for mask generation.
    """

    def __init__(
            self,
            concepts_list,
            tokenizer,
            size=512,
            center_crop=False,
            hflip=False,
            aug=True,
            num_instance_images_per_identity=4
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.aug = aug
        self.instance_images_per_identity = num_instance_images_per_identity
        self.instance_images_path = []

        # Load instance images
        for concept in concepts_list:
            base_path = Path(concept["instance_data_dir"])
            identity_count = 0
            for identity_path in base_path.iterdir():
                if identity_count >= 50:
                    break
                identity_count += 1
                if identity_path.is_dir():
                    # Assuming images are in a subdirectory named 'set_B' or directly in the folder
                    image_source_dir = identity_path / "set_B"
                    if not image_source_dir.exists():
                        image_source_dir = identity_path

                    identity_name = identity_path.name
                    instance_prompt = concept["instance_prompt"]
                    selected_images = []

                    for x in image_source_dir.iterdir():
                        if x.is_file() and x.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                            if len(selected_images) < self.instance_images_per_identity:
                                selected_images.append(x)

                    # Skip if not enough images
                    if len(selected_images) < 4:
                        continue

                    inst_img_path = [(x, instance_prompt, identity_name) for x in selected_images]
                    self.instance_images_path.extend(inst_img_path)

        # Removed Class Image loading logic as it is not needed for mask generation

        random.shuffle(self.instance_images_path)
        self.num_instance_images = len(self.instance_images_path)
        self._length = self.num_instance_images
        self.flip = transforms.RandomHorizontalFlip(
            0.5 * hflip)

        self.pil_transforms = transforms.Compose(
            [
                self.flip,
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
            ]
        )
        self.tensor_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image, instance_prompt, identity_name = self.instance_images_path[index % self.num_instance_images]
        example["original_filename"] = Path(instance_image).name

        instance_image = Image.open(instance_image)
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")

        processed_pil_image = self.pil_transforms(instance_image)
        processed_pil_image = self.tensor_transforms(processed_pil_image)

        # Mask defaults to all ones (full image)
        mask = torch.ones_like(processed_pil_image)

        # Apply resize augmentation logic for prompting
        random_scale = self.size
        if self.aug:
            random_scale = (
                np.random.randint(self.size // 3, self.size + 1)
                if np.random.uniform() < 0.66
                else np.random.randint(int(1.2 * self.size), int(1.4 * self.size))
            )

        # Note: These prompt modifications affect the example dict,
        # but the main training loop currently uses a fixed global prompt.
        if random_scale < 0.6 * self.size:
            instance_prompt = np.random.choice(["a far away ", "very small "]) + instance_prompt
        elif random_scale > self.size:
            instance_prompt = np.random.choice(["zoomed in ", "close up "]) + instance_prompt

        example["instance_images"] = processed_pil_image
        example['identity_name'] = identity_name
        example['mask'] = mask

        return example


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="UEMask training script.")
    parser.add_argument("--alpha", type=float, default=5e-3, required=True, help="PGD alpha (step size).")
    parser.add_argument("--eps", type=float, default=0.15, required=True, help="PGD epsilon (perturbation budget).")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default=None, required=True)
    parser.add_argument("--revision", type=str, default=None, required=False)
    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument("--instance_data_dir", type=str, default=None)
    parser.add_argument("--instance_prompt", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--l1_lambda", type=float, default=1e-7)
    parser.add_argument("--center_crop", default=False, action="store_true")
    parser.add_argument("--max_train_steps", type=int, default=250)
    parser.add_argument("--checkpointing_steps", type=int, default=250)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--dataloader_num_workers", type=int, default=2)
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"])
    parser.add_argument("--concepts_list", type=str, default=None)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")
    parser.add_argument("--hflip", action="store_true")
    parser.add_argument("--noaug", action="store_true")

    # Removed unused args related to class images/prior preservation

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def plot_loss_curve(losses, output_dir, alpha):
    """
    Helper function to plot the training loss curve.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title(f'Training Loss Curve alpha:{alpha}')
    plt.legend()
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f'loss_curve_alpha{alpha}.png')
    plt.savefig(plot_path)
    plt.close()


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    accelerator.init_trackers("UEMask", config=vars(args))

    if args.seed is not None:
        set_seed(args.seed)

    if args.concepts_list is None:
        args.concepts_list = [
            {
                "instance_prompt": args.instance_prompt,
                "instance_data_dir": args.instance_data_dir,
            }
        ]
    else:
        with open(args.concepts_list, "r") as f:
            args.concepts_list = json.load(f)

    # Removed "Generate class images" block completely.

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    # Load Tokenizer & Models
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, revision=args.revision, use_fast=False)
    elif args.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer",
                                                  revision=args.revision, use_fast=False)

    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = text_encoder_cls.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder",
                                                    revision=args.revision)
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet",
                                                revision=args.revision)

    # Freeze models (we are optimizing the mask, not the model weights)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    text_encoder.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Dataset creation
    train_dataset = CustomDiffusionDataset(
        concepts_list=args.concepts_list,
        tokenizer=tokenizer,
        size=args.resolution,
        center_crop=args.center_crop,
        hflip=args.hflip,
        aug=not args.noaug,
    )
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    # Prepare input_ids (Global for the loop)
    input_ids = train_dataset.tokenizer(
        args.instance_prompt,
        truncation=True,
        padding="max_length",
        max_length=train_dataset.tokenizer.model_max_length,
        return_tensors="pt",
    ).input_ids.repeat(1, 1)

    logger.info("***** Running training (Mask Generation) *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    train_loss = []
    total_generation_time = 0.0
    processed_count = 0

    for index, batch in enumerate(train_dataloader):
        original_images = batch["instance_images"].to(accelerator.device)
        mask = batch['mask'].to(accelerator.device)
        identity_name = batch['identity_name'][0]
        mask_original = mask.clone()
        original_filename = batch["original_filename"][0]
        filename_base = os.path.splitext(original_filename)[0]

        # Adjust filename for saving
        save_filename = f"{filename_base}.png"

        final_save_path = os.path.join(f"{args.output_dir}/{args.max_train_steps}/{identity_name}", save_filename)
        final_mask_save_path = os.path.join(f'{args.output_dir}', f'mask_{identity_name}', f"mask_{index}.png")

        # Skip logic if files already exist
        if os.path.exists(final_save_path):
            print(f"Final output and mask exist for {save_filename}. Skipping to next image.")
            logger.info(f"Final output and mask exist for {save_filename}. Skipping to next image.")
            continue
        elif os.path.exists(final_save_path) or os.path.exists(final_mask_save_path):
            logger.warning(f"Partial files exist for {save_filename}. Re-processing.")

        print(f"--> Start processing image: {save_filename}")
        image_start_time = time.time()
        global_step = 0
        progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
        progress_bar.set_description("Steps")

        for epoch in range(args.max_train_steps):
            mask.requires_grad = True
            protected_images = original_images * mask

            # Forward pass: VAE Encode -> Noise -> UNet Predict
            latents = vae.encode(protected_images.to(accelerator.device).to(dtype=weight_dtype)).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,),
                                      device=latents.device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            encoder_hidden_states = text_encoder(input_ids.to(accelerator.device))[0]

            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            # Loss calculation
            loss_mse = F.mse_loss(model_pred.float(), target.float(), reduction="none").mean()
            l1_penalty = torch.norm(1.0 - mask, p=1, dim=(1, 2, 3)).mean()
            l1_loss = args.l1_lambda * l1_penalty
            loss = loss_mse - l1_loss

            # Backward pass (Calculate gradients for mask)
            accelerator.backward(loss)

            # PGD Attack Step
            alpha = args.alpha
            eps = args.eps
            adv_mask = mask + alpha * mask.grad.sign()
            # Projection onto the L_inf ball centered at the original mask (all ones)
            eta = torch.clamp(adv_mask - mask_original, min=-eps, max=+eps)
            # Clip to valid image range [0, 1]
            mask = torch.clamp(mask_original + eta, min=0, max=+1).detach_()

            progress_bar.update(1)
            global_step += 1

            loss_value = loss.detach().item()
            logs = {"loss": loss_value}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            train_loss.append(loss_value)

            if global_step % args.checkpointing_steps == 0:
                image_end_time = time.time()
                single_image_duration = image_end_time - image_start_time
                total_generation_time += single_image_duration
                processed_count += 1

                print("\n<------- Statistics ------->")
                if processed_count > 0:
                    average_time = total_generation_time / processed_count
                    print(f"Total Images Processed: {processed_count}")
                    print(f"Average Time Per Image: {average_time:.4f} seconds")
                print("<-------------------------->\n")

                save_folder = f"{args.output_dir}/{global_step}/{identity_name}"
                os.makedirs(save_folder, exist_ok=True)

                noised_imgs = original_images.detach() * mask
                img_pixel = noised_imgs[0]
                save_path = os.path.join(save_folder, save_filename)

                # Save the protected image
                Image.fromarray(
                    (img_pixel * 127.5 + 128).clamp(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
                ).save(save_path)
                print(f"Saved noise at step {global_step} to {save_folder}")

                # Save Mask
                mask_save_folder = os.path.join(f'{args.output_dir}', f'mask_{identity_name}')
                os.makedirs(mask_save_folder, exist_ok=True)
                mask_save_path = os.path.join(mask_save_folder, f"mask_{index}.png")

                single_channel_mask = mask[0][0]
                mask_to_save = (single_channel_mask * 255).clamp(0, 255).to(torch.uint8)
                Image.fromarray(mask_to_save.cpu().numpy()).save(mask_save_path)

                if global_step >= args.max_train_steps:
                    break

    plot_loss_dir = './loss_plot'
    plot_loss_curve(train_loss, plot_loss_dir, args.alpha)
    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
    print("<-------end-------->")