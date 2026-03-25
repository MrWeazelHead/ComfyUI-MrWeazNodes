# MrWeaz Nodes - Flux Klein QoL Nodes

import torch
import torch.nn.functional as F
import hashlib
import json
import os
import numpy as np
import nodes
import comfy.samplers
import node_helpers
import folder_paths
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from comfy.cli_args import args


# ──────────────────────────────────────────────────────────────
# Quick Edit Reference Loader
# ──────────────────────────────────────────────────────────────

class MrWeazKleinRefLoader:
    """Loads an image directly on-node and prepares minimal Klein edit conditioning."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae": ("VAE",),
                "upload_image": ("COMBO", {
                    "image_upload": True,
                    "image_folder": "input",
                    "remote": {
                        "route": "/internal/files/input",
                        "control_after_refresh": "first",
                    },
                }),
            },
            "optional": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "edit_downscale_factor": ("FLOAT", {"default": 1.0, "min": 0.25, "max": 1.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("LATENT", "CONDITIONING", "CONDITIONING", "IMAGE", "MASK")
    RETURN_NAMES = ("reference_latent", "positive", "negative", "preview_image", "preview_mask")
    FUNCTION = "prepare_edit"
    CATEGORY = "MrWeazNodes/Klein"

    def prepare_edit(
        self,
        vae,
        upload_image,
        positive=None,
        negative=None,
        edit_downscale_factor=1.0,
    ):
        image, mask = nodes.LoadImage().load_image(upload_image)
        if edit_downscale_factor < 1.0:
            scaled_width = max(16, int((image.shape[2] * edit_downscale_factor) // 8) * 8)
            scaled_height = max(16, int((image.shape[1] * edit_downscale_factor) // 8) * 8)
            image = F.interpolate(
                image.movedim(-1, 1),
                size=(scaled_height, scaled_width),
                mode="bilinear",
                align_corners=False,
            ).movedim(1, -1)
            if mask is not None and torch.count_nonzero(mask) > 0:
                if mask.dim() == 2:
                    mask = mask.unsqueeze(0)
                mask = F.interpolate(
                    mask.unsqueeze(1),
                    size=(scaled_height, scaled_width),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)
        pixels = image.clone()

        latent = vae.encode(pixels[:, :, :, :3])
        result = {"samples": latent}

        conditioned_image = pixels[:, :, :, :3]
        mask_resized = None

        if mask is not None and torch.count_nonzero(mask) > 0:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)
        else:
            mask = None

        if mask is not None:
            mask_image = F.interpolate(
                mask.unsqueeze(1),
                size=(pixels.shape[1], pixels.shape[2]),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1).clamp(0.0, 1.0)

            # For edit/inpaint conditioning, masked regions are replaced with neutral gray
            # so Klein gets the preserved context outside the editable area.
            conditioned_image = ((conditioned_image - 0.5) * (1.0 - mask_image.unsqueeze(-1))) + 0.5

            mask_resized = F.interpolate(
                mask.unsqueeze(1),
                size=(latent.shape[2], latent.shape[3]),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1).clamp(0.0, 1.0)
            result["noise_mask"] = mask_resized

        concat_latent_image = vae.encode(conditioned_image)

        if positive is not None:
            positive = node_helpers.conditioning_set_values(
                positive,
                {"concat_latent_image": concat_latent_image},
            )
            positive = node_helpers.conditioning_set_values(
                positive,
                {"reference_latents": [latent]},
                append=True,
            )
            if mask is not None:
                positive = node_helpers.conditioning_set_values(positive, {"concat_mask": mask_resized})

        if negative is not None:
            negative = node_helpers.conditioning_set_values(
                negative,
                {"concat_latent_image": concat_latent_image},
            )
            negative = node_helpers.conditioning_set_values(
                negative,
                {"reference_latents": [latent]},
                append=True,
            )
            if mask is not None:
                negative = node_helpers.conditioning_set_values(negative, {"concat_mask": mask_resized})

        if positive is None:
            positive = []
        if negative is None:
            negative = []

        preview_mask = mask if mask is not None else torch.zeros((1, pixels.shape[1], pixels.shape[2]), dtype=torch.float32)

        return (result, positive, negative, image, preview_mask)

    @classmethod
    def IS_CHANGED(s, vae, upload_image, positive=None, negative=None):
        image_path = folder_paths.get_annotated_filepath(upload_image)
        if os.path.exists(image_path):
            return os.path.getmtime(image_path)
        return ""

    @classmethod
    def VALIDATE_INPUTS(s, vae, upload_image, positive=None, negative=None):
        if not folder_paths.exists_annotated_filepath(upload_image):
            return "Invalid image file: {}".format(upload_image)
        return True


class MrWeazImageReference:
    """Single-node version of the working Klein reference chain: scale, encode, reference-condition."""

    @staticmethod
    def _append_reference_latent(conditioning, ref_latent):
        conditioning = node_helpers.conditioning_set_values(
            conditioning,
            {"reference_latents": [ref_latent]},
            append=True,
        )
        return node_helpers.conditioning_set_values(
            conditioning,
            {"reference_latents_method": "offset"},
        )

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "upload_image": ("COMBO", {
                    "image_upload": True,
                    "image_folder": "input",
                    "remote": {
                        "route": "/internal/files/input",
                        "control_after_refresh": "first",
                    },
                }),
            },
            "optional": {
                "upscale_method": (["nearest-exact", "bilinear", "area", "bicubic", "lanczos"], {"default": "nearest-exact"}),
                "megapixels": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 16.0, "step": 0.01}),
                "resolution_steps": ("INT", {"default": 1, "min": 1, "max": 256, "step": 1}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "IMAGE")
    RETURN_NAMES = ("positive", "negative", "preview_image")
    FUNCTION = "apply_reference"
    CATEGORY = "MrWeazNodes/Klein"

    def apply_reference(
        self,
        positive,
        negative,
        vae,
        upload_image,
        upscale_method="nearest-exact",
        megapixels=1.0,
        resolution_steps=1,
    ):
        image, _ = nodes.LoadImage().load_image(upload_image)
        samples = image[:, :, :, :3].movedim(-1, 1)
        total = megapixels * 1024 * 1024
        scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
        width = round(samples.shape[3] * scale_by / resolution_steps) * resolution_steps
        height = round(samples.shape[2] * scale_by / resolution_steps) * resolution_steps
        scaled = comfy.utils.common_upscale(samples, int(width), int(height), upscale_method, "disabled").movedim(1, -1)
        latent = vae.encode(scaled)
        positive = self._append_reference_latent(positive, latent)
        negative = self._append_reference_latent(negative, latent)
        return (positive, negative, scaled)

    @classmethod
    def IS_CHANGED(
        s,
        positive,
        negative,
        vae,
        upload_image,
        upscale_method="nearest-exact",
        megapixels=1.0,
        resolution_steps=1,
    ):
        image_path = folder_paths.get_annotated_filepath(upload_image)
        mtime = os.path.getmtime(image_path) if os.path.exists(image_path) else ""
        return f"{mtime}_{upscale_method}_{megapixels}_{resolution_steps}"

    @classmethod
    def VALIDATE_INPUTS(
        s,
        positive,
        negative,
        vae,
        upload_image,
        upscale_method="nearest-exact",
        megapixels=1.0,
        resolution_steps=1,
    ):
        if not folder_paths.exists_annotated_filepath(upload_image):
            return "Invalid image file: {}".format(upload_image)
        return True

# ──────────────────────────────────────────────────────────────
# Prompt Blender
# ──────────────────────────────────────────────────────────────

class MrWeazPromptBlender:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "subject": ("STRING", {"multiline": True, "placeholder": "e.g. A beautiful woman..."}),
                "action_or_scenario": ("STRING", {"multiline": True, "placeholder": "e.g. drinking coffee"}),
                "environment": ("STRING", {"multiline": True, "placeholder": "e.g. in a bustling cyberpunk cafe"}),
                "style_and_lighting": ("STRING", {"multiline": True, "placeholder": "e.g. cinematic lighting, 8k, masterpiece"}),
                "separator": ("STRING", {"default": ", "})
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "process"
    CATEGORY = "MrWeazNodes/Klein"

    def process(self, subject, action_or_scenario, environment, style_and_lighting, separator):
        parts = [subject, action_or_scenario, environment, style_and_lighting]
        # Filter out empty strings
        parts = [p.strip() for p in parts if p and p.strip() != ""]
        final_prompt = separator.join(parts)
        return (final_prompt,)


# ──────────────────────────────────────────────────────────────
# Grid Stitcher
# ──────────────────────────────────────────────────────────────

class MrWeazGridStitcher:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "orientation": (["Side-by-Side (Horizontal)", "Top-and-Bottom (Vertical)"], {"default": "Side-by-Side (Horizontal)"})
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("grid_image",)
    FUNCTION = "process"
    CATEGORY = "MrWeazNodes/Image"

    def process(self, image1, image2, orientation):
        b1, h1, w1, c1 = image1.shape
        b2, h2, w2, c2 = image2.shape
        
        img1_p = image1.permute(0, 3, 1, 2)
        img2_p = image2.permute(0, 3, 1, 2)
        
        max_b = max(b1, b2)
        if b1 < max_b:
            img1_p = img1_p.repeat(max_b, 1, 1, 1)
        if b2 < max_b:
            img2_p = img2_p.repeat(max_b, 1, 1, 1)
        
        if orientation == "Side-by-Side (Horizontal)":
            if h1 != h2:
                new_w2 = int(w2 * (h1 / h2))
                img2_p = F.interpolate(img2_p, size=(h1, new_w2), mode="bilinear", align_corners=False)
            stitched = torch.cat((img1_p, img2_p), dim=3)
        else:
            if w1 != w2:
                new_h2 = int(h2 * (w1 / w2))
                img2_p = F.interpolate(img2_p, size=(new_h2, w1), mode="bilinear", align_corners=False)
            stitched = torch.cat((img1_p, img2_p), dim=2)
            
        return (stitched.permute(0, 2, 3, 1),)

# ──────────────────────────────────────────────────────────────
# Mask Feather & Expand
# ──────────────────────────────────────────────────────────────

class MrWeazMaskFeatherExpand:
    """Expands (dilates) and feathers a mask for inpainting/editing."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
                "expand_pixels": ("INT", {"default": 8, "min": 0, "max": 256, "step": 1}),
                "feather_radius": ("INT", {"default": 16, "min": 0, "max": 256, "step": 1}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("MASK",)
    FUNCTION = "process"
    CATEGORY = "MrWeazNodes/Klein"

    def process(self, mask, expand_pixels, feather_radius):
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        result = mask.clone()

        # 1. Dilation (expand)
        if expand_pixels > 0:
            kernel_size = expand_pixels * 2 + 1
            # Use max-pool for dilation
            result = result.unsqueeze(1)  # (B, 1, H, W)
            result = F.max_pool2d(result, kernel_size=kernel_size, stride=1,
                                  padding=expand_pixels)
            result = result.squeeze(1)  # (B, H, W)

        # 2. Gaussian feather (blur)
        if feather_radius > 0:
            kernel_size = feather_radius * 2 + 1
            sigma = feather_radius / 3.0

            # Build 1D Gaussian kernel
            x = torch.arange(kernel_size, dtype=torch.float32, device=result.device) - feather_radius
            gauss_1d = torch.exp(-0.5 * (x / sigma) ** 2)
            gauss_1d = gauss_1d / gauss_1d.sum()

            # Separable 2D Gaussian via two 1D passes
            result = result.unsqueeze(1)  # (B, 1, H, W)

            # Horizontal pass
            k_h = gauss_1d.view(1, 1, 1, kernel_size)
            result = F.pad(result, (feather_radius, feather_radius, 0, 0), mode='replicate')
            result = F.conv2d(result, k_h)

            # Vertical pass
            k_v = gauss_1d.view(1, 1, kernel_size, 1)
            result = F.pad(result, (0, 0, feather_radius, feather_radius), mode='replicate')
            result = F.conv2d(result, k_v)

            result = result.squeeze(1)  # (B, H, W)

        result = torch.clamp(result, 0.0, 1.0)
        return (result,)


# ──────────────────────────────────────────────────────────────
# Klein Batch Runner (Seed Generator)
# ──────────────────────────────────────────────────────────────

class MrWeazKleinBatchSeeds:
    """Generates a list of seeds for batch variation runs."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "base_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "batch_count": ("INT", {"default": 4, "min": 1, "max": 64}),
                "mode": (["Sequential", "Random"],),
            }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("seeds",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "generate"
    CATEGORY = "MrWeazNodes/Klein"

    def generate(self, base_seed, batch_count, mode):
        seeds = []
        if mode == "Sequential":
            for i in range(batch_count):
                seeds.append(base_seed + i)
        else:  # Random – deterministic from base_seed
            for i in range(batch_count):
                # Hash-based deterministic derivation
                h = hashlib.sha256(f"{base_seed}_{i}".encode()).hexdigest()
                derived = int(h[:16], 16)  # 64-bit seed
                seeds.append(derived)
        return (seeds,)


# ──────────────────────────────────────────────────────────────
# Image Comparer (Pro)
# ──────────────────────────────────────────────────────────────

class MrWeazImageComparer(nodes.PreviewImage):
    """A professional image comparer node with a custom UI widget."""

    def __init__(self):
        super().__init__()

    @staticmethod
    def _safe_output_dir(subfolder):
        base_output = os.path.abspath(folder_paths.get_output_directory())
        cleaned = (subfolder or "").strip().replace("/", os.sep).replace("\\", os.sep)
        full_output = os.path.abspath(os.path.join(base_output, cleaned))
        if not full_output.startswith(base_output):
            raise ValueError("custom_subfolder must stay inside the ComfyUI output directory.")
        os.makedirs(full_output, exist_ok=True)
        return full_output

    @staticmethod
    def _build_metadata(prompt, extra_pnginfo, include_metadata, compare_meta=None):
        if not include_metadata or args.disable_metadata:
            return None

        metadata = PngInfo()
        if prompt is not None:
            metadata.add_text("prompt", json.dumps(prompt))
        if extra_pnginfo is not None:
            for key in extra_pnginfo:
                metadata.add_text(key, json.dumps(extra_pnginfo[key]))
        if compare_meta is not None:
            for key, value in compare_meta.items():
                metadata.add_text(key, json.dumps(value))
        return metadata

    def _save_persistent_images(self, images, filename_prefix, custom_subfolder, prompt=None, extra_pnginfo=None, include_metadata=True, compare_meta=None):
        output_dir = self._safe_output_dir(custom_subfolder)
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix, output_dir, images[0].shape[1], images[0].shape[0]
        )

        results = []
        for batch_number, image in enumerate(images):
            arr = 255.0 * image.cpu().numpy()
            img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
            metadata = self._build_metadata(prompt, extra_pnginfo, include_metadata, compare_meta=compare_meta)
            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}_.png"
            img.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=self.compress_level)
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": "output",
            })
            counter += 1
        return results

    @staticmethod
    def _stitch_compare_image(image_a, image_b, layout):
        if image_a is None:
            return image_b
        if image_b is None:
            return image_a

        b1, h1, w1, _ = image_a.shape
        b2, h2, w2, _ = image_b.shape

        img1 = image_a.permute(0, 3, 1, 2)
        img2 = image_b.permute(0, 3, 1, 2)

        batch = max(b1, b2)
        if b1 < batch:
            img1 = img1.repeat(batch, 1, 1, 1)
        if b2 < batch:
            img2 = img2.repeat(batch, 1, 1, 1)

        if layout == "Top-Bottom":
            if w1 != w2:
                new_h2 = int(h2 * (w1 / w2))
                img2 = F.interpolate(img2, size=(new_h2, w1), mode="bilinear", align_corners=False)
            stitched = torch.cat((img1, img2), dim=2)
        else:
            if h1 != h2:
                new_w2 = int(w2 * (h1 / h2))
                img2 = F.interpolate(img2, size=(h1, new_w2), mode="bilinear", align_corners=False)
            stitched = torch.cat((img1, img2), dim=3)

        return stitched.permute(0, 2, 3, 1)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "enable_persistent_save": ("BOOLEAN", {"default": False}),
                "filename_prefix": ("STRING", {"default": "mrweaz/compare"}),
                "custom_subfolder": ("STRING", {"default": "mrweaz_compare"}),
                "save_image_a": ("BOOLEAN", {"default": True}),
                "save_image_b": ("BOOLEAN", {"default": True}),
                "save_stitched_compare": ("BOOLEAN", {"default": True}),
                "stitched_layout": (["Side-by-Side", "Top-Bottom"], {"default": "Side-by-Side"}),
                "include_metadata": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "compare_images"
    OUTPUT_NODE = True
    CATEGORY = "MrWeazNodes/Image"

    def compare_images(
        self,
        enable_persistent_save,
        filename_prefix,
        custom_subfolder,
        save_image_a,
        save_image_b,
        save_stitched_compare,
        stitched_layout,
        include_metadata,
        image_a=None,
        image_b=None,
        prompt=None,
        extra_pnginfo=None,
    ):
        # We output to UI directly
        result = {"ui": {"a_images": [], "b_images": [], "saved_images": []}}
        
        if image_a is not None:
            # save_images is a helper from nodes.PreviewImage
            saved_a = self.save_images(image_a, filename_prefix="mrweaz.compare.a", prompt=prompt, extra_pnginfo=extra_pnginfo)
            result["ui"]["a_images"] = saved_a["ui"]["images"]
            
        if image_b is not None:
            saved_b = self.save_images(image_b, filename_prefix="mrweaz.compare.b", prompt=prompt, extra_pnginfo=extra_pnginfo)
            result["ui"]["b_images"] = saved_b["ui"]["images"]

        if enable_persistent_save and (image_a is not None or image_b is not None):
            compare_meta = {
                "comparer_node": "MrWeazImageComparer",
                "stitched_layout": stitched_layout,
                "has_image_a": image_a is not None,
                "has_image_b": image_b is not None,
            }

            if save_image_a and image_a is not None:
                result["ui"]["saved_images"] += self._save_persistent_images(
                    image_a,
                    f"{filename_prefix}_A",
                    custom_subfolder,
                    prompt=prompt,
                    extra_pnginfo=extra_pnginfo,
                    include_metadata=include_metadata,
                    compare_meta=compare_meta,
                )
            if save_image_b and image_b is not None:
                result["ui"]["saved_images"] += self._save_persistent_images(
                    image_b,
                    f"{filename_prefix}_B",
                    custom_subfolder,
                    prompt=prompt,
                    extra_pnginfo=extra_pnginfo,
                    include_metadata=include_metadata,
                    compare_meta=compare_meta,
                )
            if save_stitched_compare:
                stitched = self._stitch_compare_image(image_a, image_b, stitched_layout)
                if stitched is not None:
                    result["ui"]["saved_images"] += self._save_persistent_images(
                        stitched,
                        f"{filename_prefix}_compare",
                        custom_subfolder,
                        prompt=prompt,
                        extra_pnginfo=extra_pnginfo,
                        include_metadata=include_metadata,
                        compare_meta=compare_meta,
                    )

        return result





# ──────────────────────────────────────────────────────────────
# Registration
# ──────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "MrWeazKleinRefLoader": MrWeazKleinRefLoader,
    "MrWeazMaskFeatherExpand": MrWeazMaskFeatherExpand,
    "MrWeazKleinBatchSeeds": MrWeazKleinBatchSeeds,
    "MrWeazPromptBlender": MrWeazPromptBlender,
    "MrWeazGridStitcher": MrWeazGridStitcher,
    "MrWeazImageComparer": MrWeazImageComparer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MrWeazKleinRefLoader": "(MRW) Advanced Klein Edit",
    "MrWeazMaskFeatherExpand": "(MRW) Mask Feather & Expand",
    "MrWeazKleinBatchSeeds": "(MRW) Batch Seed Runner",
    "MrWeazPromptBlender": "(MRW) Prompt Blender",
    "MrWeazGridStitcher": "(MRW) Reference Grid Stitcher",
    "MrWeazImageComparer": "(MRW) Pro Image Comparer",
}
