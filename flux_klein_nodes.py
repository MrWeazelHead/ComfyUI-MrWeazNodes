# MrWeaz Nodes - Flux Klein QoL Nodes

import torch
import torch.nn.functional as F
import hashlib
import nodes
import comfy.samplers


# ──────────────────────────────────────────────────────────────
# Quick Edit Reference Loader
# ──────────────────────────────────────────────────────────────

class MrWeazKleinRefLoader:
    """Encodes an image into a latent for Flux Klein editing workflows."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "vae": ("VAE",),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("reference_latent",)
    FUNCTION = "encode"
    CATEGORY = "MrWeazNodes/Klein"

    def encode(self, image, vae, mask=None):
        # VAE expects (B, C, H, W) but ComfyUI images are (B, H, W, C)
        pixels = image.clone()

        # Encode through VAE
        latent = vae.encode(pixels[:, :, :, :3])

        result = {"samples": latent}

        if mask is not None:
            # Resize mask to latent dimensions
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)
            _, lh, lw = latent.shape[0], latent.shape[2], latent.shape[3]
            mask_resized = F.interpolate(mask.unsqueeze(1), size=(lh, lw), mode="bilinear", align_corners=False)
            mask_resized = mask_resized.squeeze(1)
            result["noise_mask"] = mask_resized

        return (result,)

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

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
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

    def compare_images(self, image_a=None, image_b=None, prompt=None, extra_pnginfo=None):
        # We output to UI directly
        result = {"ui": {"a_images": [], "b_images": []}}
        
        if image_a is not None:
            # save_images is a helper from nodes.PreviewImage
            saved_a = self.save_images(image_a, filename_prefix="mrweaz.compare.a", prompt=prompt, extra_pnginfo=extra_pnginfo)
            result["ui"]["a_images"] = saved_a["ui"]["images"]
            
        if image_b is not None:
            saved_b = self.save_images(image_b, filename_prefix="mrweaz.compare.b", prompt=prompt, extra_pnginfo=extra_pnginfo)
            result["ui"]["b_images"] = saved_b["ui"]["images"]
            
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
    "MrWeazKleinRefLoader": "(MRW) Klein Quick Edit Ref Loader",
    "MrWeazMaskFeatherExpand": "(MRW) Mask Feather & Expand",
    "MrWeazKleinBatchSeeds": "(MRW) Batch Seed Runner",
    "MrWeazPromptBlender": "(MRW) Prompt Blender",
    "MrWeazGridStitcher": "(MRW) Reference Grid Stitcher",
    "MrWeazImageComparer": "(MRW) Pro Image Comparer",
}

