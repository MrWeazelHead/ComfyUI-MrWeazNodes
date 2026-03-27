# MrWeaz Nodes - Flux Klein QoL Nodes

import torch
import torch.nn.functional as F
import hashlib
import json
import os
import math
import fnmatch
import numpy as np
import nodes
import comfy.samplers
import node_helpers
import folder_paths
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from comfy.cli_args import args


# ──────────────────────────────────────────────────────────────
# Klein Edit Passer Node
# ──────────────────────────────────────────────────────────────

class MrWeazKleinEdit:
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
                "edit_downscale_factor": ("FLOAT", {"default": 1.0, "min": 0.25, "max": 1.0, "step": 0.05}),
                "auto_fill_mask": ("BOOLEAN", {"default": False}),
                "auto_expand_pixels": ("INT", {"default": 0, "min": 0, "max": 256, "step": 1}),
                "auto_feather_radius": ("INT", {"default": 0, "min": 0, "max": 256, "step": 1}),
            },
            "optional": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
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
        edit_downscale_factor=1.0,
        auto_fill_mask=False,
        auto_expand_pixels=0,
        auto_feather_radius=0,
        positive=None,
        negative=None,
    ):
        image, mask = nodes.LoadImage().load_image(upload_image)
        if edit_downscale_factor is None:
            edit_downscale_factor = 1.0
        try:
            edit_downscale_factor = float(edit_downscale_factor)
        except (TypeError, ValueError):
            edit_downscale_factor = 1.0
        if not math.isfinite(edit_downscale_factor):
            edit_downscale_factor = 1.0
        edit_downscale_factor = max(0.25, min(1.0, edit_downscale_factor))

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
            mask = self._process_mask(mask, auto_fill_mask, auto_expand_pixels, auto_feather_radius)
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

    @staticmethod
    def _process_mask(mask, fill_mask, expand_pixels, feather_radius):
        result = mask.clone()

        if fill_mask:
            result = (result > 0).to(result.dtype)

        if expand_pixels > 0:
            kernel_size = expand_pixels * 2 + 1
            result = result.unsqueeze(1)
            result = F.max_pool2d(result, kernel_size=kernel_size, stride=1, padding=expand_pixels)
            result = result.squeeze(1)

        if feather_radius > 0:
            kernel_size = feather_radius * 2 + 1
            sigma = feather_radius / 3.0
            x = torch.arange(kernel_size, dtype=torch.float32, device=result.device) - feather_radius
            gauss_1d = torch.exp(-0.5 * (x / sigma) ** 2)
            gauss_1d = gauss_1d / gauss_1d.sum()

            result = result.unsqueeze(1)

            k_h = gauss_1d.view(1, 1, 1, kernel_size)
            result = F.pad(result, (feather_radius, feather_radius, 0, 0), mode='replicate')
            result = F.conv2d(result, k_h)

            k_v = gauss_1d.view(1, 1, kernel_size, 1)
            result = F.pad(result, (0, 0, feather_radius, feather_radius), mode='replicate')
            result = F.conv2d(result, k_v)

            result = result.squeeze(1)

        return torch.clamp(result, 0.0, 1.0)

    @classmethod
    def IS_CHANGED(s, vae, upload_image, edit_downscale_factor=1.0, auto_fill_mask=False, auto_expand_pixels=0, auto_feather_radius=0, positive=None, negative=None):
        image_path = folder_paths.get_annotated_filepath(upload_image)
        if os.path.exists(image_path):
            return os.path.getmtime(image_path)
        return ""

    @classmethod
    def VALIDATE_INPUTS(s, vae, upload_image, edit_downscale_factor=1.0, auto_fill_mask=False, auto_expand_pixels=0, auto_feather_radius=0, positive=None, negative=None):
        if not folder_paths.exists_annotated_filepath(upload_image):
            return "Invalid image file: {}".format(upload_image)
        return True

# ──────────────────────────────────────────────────────────────
# Prompt Blender
# ──────────────────────────────────────────────────────────────

class MrWeazPromptBlender:
    """Quick & easy prompt creator."""
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
    """Stitches two images together, useful for creating A/B comparisons."""
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
                "fill_mask": ("BOOLEAN", {"default": False}),
                "expand_pixels": ("INT", {"default": 8, "min": 0, "max": 256, "step": 1}),
                "feather_radius": ("INT", {"default": 16, "min": 0, "max": 256, "step": 1}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("MASK",)
    FUNCTION = "process"
    CATEGORY = "MrWeazNodes/Klein"

    def process(self, mask, fill_mask, expand_pixels, feather_radius):
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        result = mask.clone()

        if fill_mask:
            result = (result > 0).to(result.dtype)

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
    def _extract_node(prompt, node_ref):
        if not isinstance(prompt, dict):
            return None
        node_id = node_ref
        if isinstance(node_ref, (list, tuple)) and len(node_ref) > 0:
            node_id = node_ref[0]
        if node_id is None:
            return None
        return prompt.get(str(node_id)) or prompt.get(node_id)

    @classmethod
    def _resolve_text_from_link(cls, prompt, node_ref, depth=0):
        if depth > 8:
            return None
        node = cls._extract_node(prompt, node_ref)
        if not isinstance(node, dict):
            return None
        node_type = str(node.get("class_type", ""))
        inputs = node.get("inputs", {}) if isinstance(node.get("inputs", {}), dict) else {}

        if "CLIPTextEncode" in node_type:
            text = inputs.get("text")
            if isinstance(text, str) and text.strip():
                return text.strip()
            return None

        for key in ("conditioning", "conditioning_to", "conditioning_from", "positive", "negative"):
            if key in inputs:
                txt = cls._resolve_text_from_link(prompt, inputs.get(key), depth + 1)
                if txt:
                    return txt
        return None

    @classmethod
    def _find_sampler_node(cls, prompt):
        if not isinstance(prompt, dict):
            return None
        sampler_ids = []
        for node_id, node in prompt.items():
            if not isinstance(node, dict):
                continue
            node_type = str(node.get("class_type", ""))
            if node_type in ("KSampler", "KSamplerAdvanced"):
                try:
                    sampler_ids.append((int(node_id), node))
                except Exception:
                    sampler_ids.append((-1, node))
        if not sampler_ids:
            return None
        sampler_ids.sort(key=lambda x: x[0], reverse=True)
        return sampler_ids[0][1]

    @classmethod
    def _build_parameters_text(cls, prompt, image_size=None):
        sampler = cls._find_sampler_node(prompt)
        if sampler is None:
            return None

        inputs = sampler.get("inputs", {}) if isinstance(sampler.get("inputs", {}), dict) else {}
        steps = inputs.get("steps")
        cfg = inputs.get("cfg")
        sampler_name = inputs.get("sampler_name")
        scheduler = inputs.get("scheduler")
        seed = inputs.get("seed")
        denoise = inputs.get("denoise")

        positive_text = cls._resolve_text_from_link(prompt, inputs.get("positive"))
        negative_text = cls._resolve_text_from_link(prompt, inputs.get("negative"))

        lines = []
        if positive_text:
            lines.append(positive_text)
        if negative_text:
            lines.append(f"Negative prompt: {negative_text}")

        param_parts = []
        if steps is not None:
            param_parts.append(f"Steps: {steps}")
        if sampler_name is not None:
            if scheduler is not None:
                param_parts.append(f"Sampler: {sampler_name} ({scheduler})")
            else:
                param_parts.append(f"Sampler: {sampler_name}")
        elif scheduler is not None:
            param_parts.append(f"Scheduler: {scheduler}")
        if cfg is not None:
            param_parts.append(f"CFG scale: {cfg}")
        if seed is not None:
            param_parts.append(f"Seed: {seed}")
        if denoise is not None:
            param_parts.append(f"Denoise: {denoise}")
        if image_size is not None and len(image_size) == 2:
            param_parts.append(f"Size: {image_size[0]}x{image_size[1]}")

        if param_parts:
            lines.append(", ".join(param_parts))

        if not lines:
            return None
        return "\n".join(lines)

    @classmethod
    def _build_metadata(cls, prompt, extra_pnginfo, include_metadata, compare_meta=None, image_size=None):
        if not include_metadata or args.disable_metadata:
            return None

        metadata = PngInfo()
        if prompt is not None:
            metadata.add_text("prompt", json.dumps(prompt))
            parameters = cls._build_parameters_text(prompt, image_size=image_size)
            if parameters:
                metadata.add_text("parameters", parameters)
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
            image_size = (int(image.shape[1]), int(image.shape[0]))
            metadata = self._build_metadata(
                prompt,
                extra_pnginfo,
                include_metadata,
                compare_meta=compare_meta,
                image_size=image_size,
            )
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
                "save_image": ("BOOLEAN", {"default": True}),
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
        save_image,
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

        if save_image and (image_a is not None or image_b is not None):
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
# Reference Latent 
# ──────────────────────────────────────────────────────────────


class MrWeazReferenceFromImage:
    """Reference encode and conditioning in one node."""
    @classmethod
    def _list_input_images(cls):
        input_dir = folder_paths.get_input_directory()
        exclude_files = {"Thumbs.db", "*.DS_Store", "desktop.ini", "*.lock"}
        exclude_folders = {"clipspace", ".*"}

        file_list = []
        for root, dirs, files in os.walk(input_dir, followlinks=True):
            dirs[:] = [d for d in dirs if not any(fnmatch.fnmatch(d, pat) for pat in exclude_folders)]
            files = [f for f in files if not any(fnmatch.fnmatch(f, pat) for pat in exclude_files)]
            for file in files:
                relpath = os.path.relpath(os.path.join(root, file), start=input_dir)
                file_list.append(relpath.replace("\\", "/"))
        return sorted(file_list)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",),
                "positive": ("CONDITIONING",),
                "upload_image": (cls._list_input_images(), {"image_upload": True}),
            },
            "optional": {
                "negative": ("CONDITIONING",),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "apply_reference"
    CATEGORY = "MrWeazNodes/Conditioning"

    @staticmethod
    def _zero_conditioning(conditioning):
        if conditioning is None:
            return []

        zeroed = []
        for entry in conditioning:
            if not isinstance(entry, (list, tuple)) or len(entry) < 2:
                continue
            cond_tensor = entry[0]
            cond_meta = dict(entry[1]) if isinstance(entry[1], dict) else {}
            cond_tensor = torch.zeros_like(cond_tensor)
            pooled = cond_meta.get("pooled_output")
            if pooled is not None:
                cond_meta["pooled_output"] = torch.zeros_like(pooled)
            zeroed.append([cond_tensor, cond_meta])
        return zeroed

    def apply_reference(self, positive, vae, upload_image, negative=None):
        image, _mask = nodes.LoadImage().load_image(upload_image)
        latent_samples = vae.encode(image[:, :, :, :3])
        positive = node_helpers.conditioning_set_values(
            positive,
            {"reference_latents": [latent_samples]},
            append=True,
        )
        negative = self._zero_conditioning(negative)
        return (positive, negative)


def _normalize_conditioning(conditioning):
    if conditioning is None:
        return []
    return conditioning


def _extract_reference_latents(conditioning):
    refs = []
    for entry in conditioning or []:
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            continue
        meta = entry[1] if isinstance(entry[1], dict) else {}
        value = meta.get("reference_latents")
        if isinstance(value, list):
            refs.extend(value)
    return refs


def _extract_first_meta(conditioning, key):
    for entry in conditioning or []:
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            continue
        meta = entry[1] if isinstance(entry[1], dict) else {}
        if key in meta:
            return meta.get(key)
    return None


def _clone_conditioning(conditioning):
    cloned = []
    for entry in conditioning or []:
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            continue
        cond_tensor = entry[0]
        cond_meta = dict(entry[1]) if isinstance(entry[1], dict) else {}
        cloned.append([cond_tensor, cond_meta])
    return cloned


def _conditioning_set_mask(conditioning, mask, strength, set_area_to_bounds):
    conditioned = []
    for entry in conditioning or []:
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            continue
        cond_tensor = entry[0]
        cond_meta = dict(entry[1]) if isinstance(entry[1], dict) else {}
        cond_meta["mask"] = mask
        cond_meta["mask_strength"] = float(strength)
        cond_meta["set_area_to_bounds"] = bool(set_area_to_bounds)
        conditioned.append([cond_tensor, cond_meta])
    return conditioned


class MrWeazRegionalPromptMixer:
    """Mixes regional prompt conditionings using masks with per-region strength."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "set_area_to_bounds": ("BOOLEAN", {"default": True}),
                "feather_radius": ("INT", {"default": 12, "min": 0, "max": 256, "step": 1}),
                "region_1_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "region_2_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "region_3_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
            },
            "optional": {
                "negative": ("CONDITIONING",),
                "region_1_positive": ("CONDITIONING",),
                "region_1_mask": ("MASK",),
                "region_2_positive": ("CONDITIONING",),
                "region_2_mask": ("MASK",),
                "region_3_positive": ("CONDITIONING",),
                "region_3_mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "STRING")
    RETURN_NAMES = ("positive", "negative", "mix_report")
    FUNCTION = "mix"
    CATEGORY = "MrWeazNodes/Conditioning"

    @staticmethod
    def _prepare_mask(mask, feather_radius):
        if mask is None:
            return None

        m = mask
        if m.dim() == 2:
            m = m.unsqueeze(0)
        elif m.dim() == 4 and m.shape[-1] == 1:
            m = m.squeeze(-1)
        elif m.dim() == 4 and m.shape[1] == 1:
            m = m.squeeze(1)

        if m.dim() != 3:
            raise ValueError(f"Unsupported mask shape {tuple(m.shape)}; expected [B,H,W] or [H,W].")

        m = torch.clamp(m, 0.0, 1.0)

        if feather_radius <= 0:
            return m

        kernel_size = feather_radius * 2 + 1
        sigma = max(0.001, feather_radius / 3.0)
        x = torch.arange(kernel_size, dtype=torch.float32, device=m.device) - feather_radius
        gauss_1d = torch.exp(-0.5 * (x / sigma) ** 2)
        gauss_1d = gauss_1d / gauss_1d.sum()

        m = m.unsqueeze(1)
        k_h = gauss_1d.view(1, 1, 1, kernel_size)
        m = F.pad(m, (feather_radius, feather_radius, 0, 0), mode="replicate")
        m = F.conv2d(m, k_h)

        k_v = gauss_1d.view(1, 1, kernel_size, 1)
        m = F.pad(m, (0, 0, feather_radius, feather_radius), mode="replicate")
        m = F.conv2d(m, k_v)
        m = m.squeeze(1)

        return torch.clamp(m, 0.0, 1.0)

    def mix(
        self,
        positive,
        set_area_to_bounds,
        feather_radius,
        region_1_strength,
        region_2_strength,
        region_3_strength,
        negative=None,
        region_1_positive=None,
        region_1_mask=None,
        region_2_positive=None,
        region_2_mask=None,
        region_3_positive=None,
        region_3_mask=None,
    ):
        mixed_positive = _clone_conditioning(positive)

        regions = [
            (1, region_1_positive, region_1_mask, float(region_1_strength)),
            (2, region_2_positive, region_2_mask, float(region_2_strength)),
            (3, region_3_positive, region_3_mask, float(region_3_strength)),
        ]

        applied_regions = []
        for idx, region_positive, region_mask, region_strength in regions:
            if region_positive is None or region_mask is None or region_strength <= 0.0:
                continue
            prepared_mask = self._prepare_mask(region_mask, int(feather_radius))
            region_conditioning = _conditioning_set_mask(
                _clone_conditioning(region_positive),
                prepared_mask,
                region_strength,
                set_area_to_bounds,
            )
            if region_conditioning:
                mixed_positive.extend(region_conditioning)
                applied_regions.append({
                    "region": idx,
                    "strength": region_strength,
                    "entries": len(region_conditioning),
                })

        mix_report = {
            "base_entries": len(_clone_conditioning(positive)),
            "applied_regions": applied_regions,
            "total_positive_entries": len(mixed_positive),
            "set_area_to_bounds": bool(set_area_to_bounds),
            "feather_radius": int(feather_radius),
            "version": 1,
        }

        negative = _normalize_conditioning(negative)
        return (mixed_positive, negative, json.dumps(mix_report))


def _fusion_mode_multipliers(fusion_mode):
    if fusion_mode == "Identity Priority":
        return {"identity": 1.35, "style": 0.8, "composition": 0.85}
    if fusion_mode == "Style Priority":
        return {"identity": 0.8, "style": 1.35, "composition": 0.85}
    if fusion_mode == "Composition Priority":
        return {"identity": 0.85, "style": 0.8, "composition": 1.35}
    return {"identity": 1.0, "style": 1.0, "composition": 1.0}


def _effective_fusion_weight(base_weight, fusion_mode, role):
    multipliers = _fusion_mode_multipliers(fusion_mode)
    return max(0.0, float(base_weight) * float(multipliers.get(role, 1.0)))


def _weighted_reference_latents(refs, effective_weight):
    if not refs or effective_weight <= 0.0:
        return []
    # Repeat references based on effective weight so downstream consumers receive a
    # practical bias without requiring custom sampler support.
    repeat_count = int(round(effective_weight * 4.0))
    repeat_count = max(1, min(8, repeat_count))
    weighted = []
    for _ in range(repeat_count):
        weighted.extend(refs)
    return weighted


def _pick_primary_conditioning_source(fusion_mode, role_data):
    # role_data: {role: {"conditioning": ..., "effective_weight": ..., "refs": ...}}
    available = []
    for role, data in role_data.items():
        conditioning = data.get("conditioning")
        if conditioning is None:
            continue
        concat_latent = _extract_first_meta(conditioning, "concat_latent_image")
        concat_mask = _extract_first_meta(conditioning, "concat_mask")
        if concat_latent is None and concat_mask is None:
            continue
        available.append({
            "role": role,
            "conditioning": conditioning,
            "effective_weight": float(data.get("effective_weight", 0.0)),
            "concat_latent": concat_latent,
            "concat_mask": concat_mask,
        })

    if not available:
        return (None, None, None)

    # Tie-break order follows the chosen fusion mode.
    if fusion_mode == "Identity Priority":
        order = {"identity": 0, "style": 1, "composition": 2}
    elif fusion_mode == "Style Priority":
        order = {"style": 0, "identity": 1, "composition": 2}
    elif fusion_mode == "Composition Priority":
        order = {"composition": 0, "identity": 1, "style": 2}
    else:
        order = {"identity": 0, "style": 1, "composition": 2}

    available.sort(key=lambda x: (-x["effective_weight"], order.get(x["role"], 99)))
    chosen = available[0]
    return (chosen["concat_latent"], chosen["concat_mask"], chosen["role"])


class MrWeazKleinContinuityController:
    """Experimental continuity state controller for Klein conditioning."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "continuity_strength": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05}),
                "style_freedom": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.05}),
                "identity_anchor": ("STRING", {"default": "", "multiline": True}),
                "lock_identity": ("BOOLEAN", {"default": True}),
                "keep_color_palette": ("BOOLEAN", {"default": True}),
                "keep_outfit_props": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "negative": ("CONDITIONING",),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "STRING")
    RETURN_NAMES = ("positive", "negative", "continuity_state")
    FUNCTION = "apply_continuity"
    CATEGORY = "MrWeazNodes/Klein/Experimental"

    def apply_continuity(
        self,
        positive,
        continuity_strength,
        style_freedom,
        identity_anchor,
        lock_identity,
        keep_color_palette,
        keep_outfit_props,
        negative=None,
    ):
        continuity_strength = max(0.0, min(1.0, float(continuity_strength)))
        style_freedom = max(0.0, min(1.0, float(style_freedom)))

        state = {
            "continuity_strength": continuity_strength,
            "style_freedom": style_freedom,
            "identity_anchor": (identity_anchor or "").strip(),
            "lock_identity": bool(lock_identity),
            "keep_color_palette": bool(keep_color_palette),
            "keep_outfit_props": bool(keep_outfit_props),
            "version": 1,
        }

        positive = node_helpers.conditioning_set_values(positive, {"mrw_klein_continuity_state": state})
        positive = node_helpers.conditioning_set_values(
            positive,
            {"mrw_klein_continuity_strength": continuity_strength},
        )
        positive = node_helpers.conditioning_set_values(
            positive,
            {"mrw_klein_style_freedom": style_freedom},
        )

        negative = _normalize_conditioning(negative)
        if negative:
            negative = node_helpers.conditioning_set_values(negative, {"mrw_klein_continuity_state": state})

        return (positive, negative, json.dumps(state))


class MrWeazKleinReferenceFusion:
    """Experimental multi-reference fusion for Klein conditioning."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "fusion_mode": (["Balanced", "Identity Priority", "Style Priority", "Composition Priority"],),
                "identity_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "style_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "composition_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
            },
            "optional": {
                "negative": ("CONDITIONING",),
                "identity_reference": ("CONDITIONING",),
                "style_reference": ("CONDITIONING",),
                "composition_reference": ("CONDITIONING",),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "STRING")
    RETURN_NAMES = ("positive", "negative", "fusion_report")
    FUNCTION = "fuse_references"
    CATEGORY = "MrWeazNodes/Klein/Experimental"

    def fuse_references(
        self,
        positive,
        fusion_mode,
        identity_weight,
        style_weight,
        composition_weight,
        negative=None,
        identity_reference=None,
        style_reference=None,
        composition_reference=None,
    ):
        identity_weight = max(0.0, min(2.0, float(identity_weight)))
        style_weight = max(0.0, min(2.0, float(style_weight)))
        composition_weight = max(0.0, min(2.0, float(composition_weight)))

        identity_refs = _extract_reference_latents(identity_reference)
        style_refs = _extract_reference_latents(style_reference)
        composition_refs = _extract_reference_latents(composition_reference)

        effective_identity_weight = _effective_fusion_weight(identity_weight, fusion_mode, "identity")
        effective_style_weight = _effective_fusion_weight(style_weight, fusion_mode, "style")
        effective_composition_weight = _effective_fusion_weight(composition_weight, fusion_mode, "composition")

        fused_refs = []
        fused_refs.extend(_weighted_reference_latents(identity_refs, effective_identity_weight))
        fused_refs.extend(_weighted_reference_latents(style_refs, effective_style_weight))
        fused_refs.extend(_weighted_reference_latents(composition_refs, effective_composition_weight))

        role_data = {
            "identity": {
                "conditioning": identity_reference,
                "effective_weight": effective_identity_weight,
                "refs": identity_refs,
            },
            "style": {
                "conditioning": style_reference,
                "effective_weight": effective_style_weight,
                "refs": style_refs,
            },
            "composition": {
                "conditioning": composition_reference,
                "effective_weight": effective_composition_weight,
                "refs": composition_refs,
            },
        }

        concat_latent, concat_mask, primary_source = _pick_primary_conditioning_source(fusion_mode, role_data)

        fusion_state = {
            "fusion_mode": fusion_mode,
            "identity_weight": identity_weight,
            "style_weight": style_weight,
            "composition_weight": composition_weight,
            "effective_identity_weight": effective_identity_weight,
            "effective_style_weight": effective_style_weight,
            "effective_composition_weight": effective_composition_weight,
            "identity_refs": len(identity_refs),
            "style_refs": len(style_refs),
            "composition_refs": len(composition_refs),
            "weighted_identity_refs": len(_weighted_reference_latents(identity_refs, effective_identity_weight)),
            "weighted_style_refs": len(_weighted_reference_latents(style_refs, effective_style_weight)),
            "weighted_composition_refs": len(_weighted_reference_latents(composition_refs, effective_composition_weight)),
            "fused_refs": len(fused_refs),
            "primary_source": primary_source,
            "version": 1,
        }

        positive = node_helpers.conditioning_set_values(positive, {"mrw_klein_fusion_state": fusion_state})
        if fused_refs:
            positive = node_helpers.conditioning_set_values(positive, {"reference_latents": fused_refs}, append=True)
        if concat_latent is not None:
            positive = node_helpers.conditioning_set_values(positive, {"concat_latent_image": concat_latent})
        if concat_mask is not None:
            positive = node_helpers.conditioning_set_values(positive, {"concat_mask": concat_mask})

        negative = _normalize_conditioning(negative)
        if negative:
            negative = node_helpers.conditioning_set_values(negative, {"mrw_klein_fusion_state": fusion_state})

        return (positive, negative, json.dumps(fusion_state))

# ──────────────────────────────────────────────────────────────
# Registration
# ──────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "MrWeazKleinEdit": MrWeazKleinEdit,
    "MrWeazMaskFeatherExpand": MrWeazMaskFeatherExpand,
    "MrWeazKleinBatchSeeds": MrWeazKleinBatchSeeds,
    "MrWeazPromptBlender": MrWeazPromptBlender,
    "MrWeazGridStitcher": MrWeazGridStitcher,
    "MrWeazImageComparer": MrWeazImageComparer,
    "MrWeazReferenceFromImage": MrWeazReferenceFromImage,
    "MrWeazRegionalPromptMixer": MrWeazRegionalPromptMixer,
    "MrWeazKleinContinuityController": MrWeazKleinContinuityController,
    "MrWeazKleinReferenceFusion": MrWeazKleinReferenceFusion,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MrWeazKleinEdit": "(MRW) Klein Edit",
    "MrWeazMaskFeatherExpand": "(MRW) Mask Feather & Expand",
    "MrWeazKleinBatchSeeds": "(MRW) Batch Seed Runner",
    "MrWeazPromptBlender": "(MRW) Prompt Blender",
    "MrWeazGridStitcher": "(MRW) Reference Grid Stitcher",
    "MrWeazImageComparer": "(MRW) Image Comparer",
    "MrWeazReferenceFromImage": "(MRW) Image Reference",
    "MrWeazRegionalPromptMixer": "(MRW) Regional Prompt Mixer",
    "MrWeazKleinContinuityController": "(MRW) Klein Continuity Controller",
    "MrWeazKleinReferenceFusion": "(MRW) Klein Reference Fusion",
}
