#MrWeaz Nodes - (MRW) Image Effects

import torch
import torch.nn.functional as F
import math
import nodes
import folder_paths
from server import PromptServer

class MrWeazCRTVHS:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "scanline_intensity": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.05}),
                "rgb_shift_amount": ("FLOAT", {"default": 0.005, "min": 0.0, "max": 0.1, "step": 0.001}),
                "noise_amount": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.05}),
                "aberration_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.1, "step": 0.005}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "MrWeazNodes/Effects"

    def process(self, image, scanline_intensity, rgb_shift_amount, noise_amount, aberration_strength=0.0):
        batch, height, width, channels = image.shape
        img = image.clone()
        
        # 1. RGB Shift (Chromatic Aberration)
        if rgb_shift_amount > 0:
            shift_pixels = max(1, int(width * rgb_shift_amount))
            # Shift Red left, Blue right
            r = img[:, :, :, 0]
            g = img[:, :, :, 1]
            b = img[:, :, :, 2]
            
            r_shifted = torch.roll(r, shifts=-shift_pixels, dims=2)
            b_shifted = torch.roll(b, shifts=shift_pixels, dims=2)
            
            img[:, :, :, 0] = r_shifted
            img[:, :, :, 2] = b_shifted
            
        # 1.5 Radial Chromatic Aberration (if aberration_strength > 0)
        if aberration_strength > 0:
            # Re-use the logic from LensAberration but integrated into CRT
            b, h, w, c = img.shape
            y_coords = torch.linspace(-1, 1, h, device=img.device)
            x_coords = torch.linspace(-1, 1, w, device=img.device)
            yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
            grid = torch.stack((xx, yy), dim=-1).unsqueeze(0).repeat(b, 1, 1, 1)
            radius = torch.sqrt(xx**2 + yy**2).unsqueeze(-1).unsqueeze(0)
            
            grid_r = grid * (1.0 - aberration_strength * radius)
            grid_b = grid * (1.0 + aberration_strength * radius)
            
            img_p = img.permute(0, 3, 1, 2)
            r_distorted = F.grid_sample(img_p[:, 0:1, :, :], grid_r, align_corners=False)
            b_distorted = F.grid_sample(img_p[:, 2:3, :, :], grid_b, align_corners=False)
            
            img_p = img_p.clone()
            img_p[:, 0:1, :, :] = r_distorted
            img_p[:, 2:3, :, :] = b_distorted
            img = img_p.permute(0, 2, 3, 1)
            
        # 2. Scanlines
        if scanline_intensity > 0:
            y = torch.arange(height, device=img.device).view(1, height, 1, 1).float()
            # 1 on even lines, 0 on odd lines
            scanlines = (y % 2)
            # Multiplier: reduces brightness of odd lines by scanline_intensity
            img = img * (1.0 - (scanlines * scanline_intensity * 0.8))
            
        # 3. Static/Noise
        if noise_amount > 0:
            noise = (torch.rand_like(img) - 0.5) * noise_amount
            img = img + noise
            
        img = torch.clamp(img, 0.0, 1.0)
        return (img,)

class MrWeazFilmGrainVignette:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "vignette_intensity": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),
                "grain_amount": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.05}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "MrWeazNodes/Effects"

    def process(self, image, vignette_intensity, grain_amount):
        batch, height, width, channels = image.shape
        img = image.clone()
        
        # 1. Vignette
        if vignette_intensity > 0:
            # Create a 2D meshgrid
            y = torch.linspace(-1, 1, height, device=img.device).view(1, height, 1, 1)
            x = torch.linspace(-1, 1, width, device=img.device).view(1, 1, width, 1)
            
            # Distance from center
            radius = torch.sqrt(x**2 + y**2)
            # Smooth vignette falloff
            vignette = 1.0 - torch.clamp(radius * vignette_intensity, 0.0, 1.0)
            
            # Apply to image
            img = img * vignette
            
        # 2. Film Grain
        if grain_amount > 0:
            # Film grain tends to be a bit blurred and monochrome, blending with luminance
            noise = torch.randn(batch, height, width, 1, device=img.device) * grain_amount * 0.3
            img = img + noise
            
        img = torch.clamp(img, 0.0, 1.0)
        return (img,)

class MrWeazPosterize:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "levels": ("INT", {"default": 4, "min": 2, "max": 256, "step": 1}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "MrWeazNodes/Effects"

    def process(self, image, levels):
        img = image.clone()
        img = torch.round(img * (levels - 1)) / (levels - 1)
        img = torch.clamp(img, 0.0, 1.0)
        return (img,)

class MrWeazLetterbox:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "bar_ratio": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 0.49, "step": 0.01}),
                "bar_color": (["Black", "White"], {"default": "Black"}),
                "orientation": (["Horizontal (Top/Bottom)", "Vertical (Left/Right)"], {"default": "Horizontal (Top/Bottom)"})
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "MrWeazNodes/Effects"

    def process(self, image, bar_ratio, bar_color, orientation):
        batch, height, width, channels = image.shape
        img = image.clone()
        
        c_val = 0.0 if bar_color == "Black" else 1.0
        
        if orientation == "Horizontal (Top/Bottom)":
            bar_pixels = int(height * bar_ratio)
            if bar_pixels > 0:
                img[:, :bar_pixels, :, :] = c_val
                img[:, -bar_pixels:, :, :] = c_val
        else:
            bar_pixels = int(width * bar_ratio)
            if bar_pixels > 0:
                img[:, :, :bar_pixels, :] = c_val
                img[:, :, -bar_pixels:, :] = c_val
                
        return (img,)

import torchvision.transforms.functional as TF

class MrWeazColorAdjust:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "brightness": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05}),
                "contrast": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05}),
                "saturation": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05}),
                "hue_shift": ("FLOAT", {"default": 0.0, "min": -0.5, "max": 0.5, "step": 0.01}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "MrWeazNodes/Effects"

    def process(self, image, brightness, contrast, saturation, hue_shift):
        img_permuted = image.permute(0, 3, 1, 2)
        
        if brightness != 1.0:
            img_permuted = TF.adjust_brightness(img_permuted, brightness)
        if contrast != 1.0:
            img_permuted = TF.adjust_contrast(img_permuted, contrast)
        if saturation != 1.0:
            img_permuted = TF.adjust_saturation(img_permuted, saturation)
        if hue_shift != 0.0:
            img_permuted = TF.adjust_hue(img_permuted, hue_shift)
            
        result = img_permuted.permute(0, 2, 3, 1)
        result = torch.clamp(result, 0.0, 1.0)
        return (result,)

class MrWeazVibeTransfer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "target_image": ("IMAGE",),
                "reference_image": ("IMAGE",),
                "blend_factor": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "MrWeazNodes/Effects"

    def process(self, target_image, reference_image, blend_factor):
        if blend_factor == 0.0:
            return (target_image,)
            
        ref_b, ref_h, ref_w, ref_c = reference_image.shape
        ref_mean = torch.mean(reference_image, dim=(1, 2), keepdim=True)
        
        tar_b = target_image.shape[0]
        if ref_b < tar_b:
            ref_mean = ref_mean.repeat(tar_b, 1, 1, 1)
        
        tar_luma = 0.299 * target_image[..., 0:1] + 0.587 * target_image[..., 1:2] + 0.114 * target_image[..., 2:3]
        ref_luma = 0.299 * ref_mean[..., 0:1] + 0.587 * ref_mean[..., 1:2] + 0.114 * ref_mean[..., 2:3]
        ref_luma = torch.clamp(ref_luma, min=0.001)
        
        color_tint = ref_mean / ref_luma 
        tinted = tar_luma.unsqueeze(-1) * color_tint if tar_luma.dim() == 3 else tar_luma * color_tint
        
        result = target_image * (1.0 - blend_factor) + tinted * blend_factor
        result = torch.clamp(result, 0.0, 1.0)
        return (result,)

import torch.nn.functional as F

class MrWeazLensAberration:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "aberration_strength": ("FLOAT", {"default": 0.02, "min": 0.0, "max": 0.1, "step": 0.005}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "MrWeazNodes/Effects"

    def process(self, image, aberration_strength):
        if aberration_strength <= 0.0:
            return (image,)
            
        b, h, w, c = image.shape
        img = image.clone()
        
        y = torch.linspace(-1, 1, h, device=img.device)
        x = torch.linspace(-1, 1, w, device=img.device)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        
        grid = torch.stack((xx, yy), dim=-1).unsqueeze(0).repeat(b, 1, 1, 1)
        radius = torch.sqrt(xx**2 + yy**2).unsqueeze(-1).unsqueeze(0)
        
        grid_r = grid * (1.0 - aberration_strength * radius)
        grid_b = grid * (1.0 + aberration_strength * radius)
        
        img_p = img.permute(0, 3, 1, 2)
        
        r_p = img_p[:, 0:1, :, :]
        r_distorted = F.grid_sample(r_p, grid_r, align_corners=False)
        
        b_p = img_p[:, 2:3, :, :]
        b_distorted = F.grid_sample(b_p, grid_b, align_corners=False)
        
        img_p = img_p.clone()
        img_p[:, 0:1, :, :] = r_distorted
        img_p[:, 2:3, :, :] = b_distorted
        
        result = img_p.permute(0, 2, 3, 1)
        result = torch.clamp(result, 0.0, 1.0)
        return (result,)

# ──────────────────────────────────────────────────────────────
# ALL-IN-ONE IMAGE EFFECTS
# ──────────────────────────────────────────────────────────────

class MrWeazAIOImageEffects(nodes.PreviewImage):
    """Consolidates all image effects into a single node with a custom comparison UI."""
    
    def __init__(self):
        super().__init__()

    @staticmethod
    def _prepare_mask(mask, image):
        """
        Normalize incoming MASK tensor to shape [B, H, W, 1] matching IMAGE batch/size.
        """
        if mask is None:
            return None

        m = mask
        # Comfy masks are usually [B,H,W], but handle common variants defensively.
        if m.dim() == 2:
            m = m.unsqueeze(0)
        elif m.dim() == 4 and m.shape[-1] == 1:
            m = m.squeeze(-1)
        elif m.dim() == 4 and m.shape[1] == 1:
            m = m.squeeze(1)

        if m.dim() != 3:
            raise ValueError(f"Unsupported mask shape {tuple(m.shape)}; expected [B,H,W] or [H,W].")

        m = m.to(device=image.device, dtype=image.dtype)

        b, h, w, _ = image.shape
        mb = m.shape[0]
        if mb == 1 and b > 1:
            m = m.expand(b, -1, -1)
        elif mb != b:
            if mb < b:
                reps = math.ceil(b / mb)
                m = m.repeat(reps, 1, 1)[:b]
            else:
                m = m[:b]

        if m.shape[1] != h or m.shape[2] != w:
            m = F.interpolate(m.unsqueeze(1), size=(h, w), mode='bilinear', align_corners=False).squeeze(1)

        return torch.clamp(m, 0.0, 1.0).unsqueeze(-1)
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "effect": ([
                    "None",
                    "GaussianBlur", "Sharpen", "Invert", "Sepia", "EdgeDetection",
                    "Glitch", "TiltShift", "Halftone", "Bloom", "SplitToning",
                    "CRTVHS", "FilmGrainVignette", "Posterize", "Letterbox",
                    "ColorAdjust", "VibeTransfer", "LensAberration",
                    "Pixelate", "Solarize", "Duotone",
                    "Emboss", "VignetteOnly", "Gamma", "CrossProcess",
                    "ChromaShift", "Sketch", "Vibrance", "NightVision"
                ], {"default": "None"}),
                
                # Gaussian Blur / Bloom
                "blur_radius": ("INT", {"default": 16, "min": 1, "max": 128}),
                "sigma": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                
                # Sharpen / Bloom / SplitToning / Vibe
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1}),
                
                # Sepia / Vibe / ColorAdjust
                "intensity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                
                # Edge Detection
                "sensitivity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1}),
                
                # Glitch
                "glitch_amount": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.05}),
                "scan_distortion": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                
                # Tilt Shift
                "focus_center": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "focus_width": ("FLOAT", {"default": 0.2, "min": 0.01, "max": 1.0, "step": 0.05}),
                
                # Halftone
                "dot_size": ("INT", {"default": 4, "min": 1, "max": 24}),
                "angle": ("FLOAT", {"default": 45.0, "min": 0, "max": 360}),
                "monochrome": (["False", "True"], {"default": "False"}),
                
                # Bloom
                "threshold": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05}),
                
                # Split Toning
                "highlight_color": ("COLOR", {"default": "#FFCC00"}),
                "shadow_color": ("COLOR", {"default": "#003366"}),
                "balance": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                
                # CRT VHS
                "scanline_intensity": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.05}),
                "rgb_shift_amount": ("FLOAT", {"default": 0.005, "min": 0.0, "max": 0.1, "step": 0.001}),
                "noise_amount": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.05}),
                
                # Film Grain
                "vignette_intensity": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),
                "grain_amount": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.05}),
                
                # Posterize
                "levels": ("INT", {"default": 4, "min": 2, "max": 256}),
                
                # Letterbox
                "bar_ratio": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 0.49, "step": 0.01}),
                "bar_color": (["Black", "White"], {"default": "Black"}),
                "orientation": (["Horizontal (Top/Bottom)", "Vertical (Left/Right)"], {"default": "Horizontal (Top/Bottom)"}),
                
                # Color Adjust
                "brightness": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05}),
                "contrast": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05}),
                "saturation": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05}),
                "hue_shift": ("FLOAT", {"default": 0.0, "min": -0.5, "max": 0.5, "step": 0.01}),
                
                # Lens Aberration
                "aberration_strength": ("FLOAT", {"default": 0.02, "min": 0.0, "max": 0.1, "step": 0.005}),
            },
            "optional": {
                "reference_image": ("IMAGE",),
                "mask": ("MASK",),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO"
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "MrWeazNodes/Effects"

    def process(self, image, effect, prompt=None, extra_pnginfo=None, **kwargs):
        result_img = image
        if effect == "None":
            result_img = image
        else:
            # Delegate to specialized classes
            if effect == "GaussianBlur":
                result_img = MrWeazGaussianBlur().process(image, kwargs.get("blur_radius", 16), kwargs.get("sigma", 1.0))[0]
            elif effect == "Sharpen":
                result_img = MrWeazSharpen().process(image, kwargs.get("strength", 1.0), kwargs.get("blur_radius", 1) // 4 + 1)[0]
            elif effect == "Invert":
                result_img = MrWeazInvert().process(image)[0]
            elif effect == "Sepia":
                result_img = MrWeazSepia().process(image, kwargs.get("intensity", 1.0))[0]
            elif effect == "EdgeDetection":
                result_img = MrWeazEdgeDetection().process(image, kwargs.get("sensitivity", 1.0))[0]
            elif effect == "Glitch":
                result_img = MrWeazGlitch().process(image, kwargs.get("glitch_amount", 0.2), kwargs.get("scan_distortion", 0.5), kwargs.get("seed", 0))[0]
            elif effect == "TiltShift":
                result_img = MrWeazTiltShift().process(image, kwargs.get("blur_radius", 16), kwargs.get("focus_center", 0.5), kwargs.get("focus_width", 0.2))[0]
            elif effect == "Halftone":
                result_img = MrWeazHalftone().process(image, kwargs.get("dot_size", 4), kwargs.get("angle", 45.0), kwargs.get("monochrome", "False"))[0]
            elif effect == "Bloom":
                result_img = MrWeazBloom().process(image, kwargs.get("threshold", 0.8), kwargs.get("strength", 0.5), kwargs.get("blur_radius", 24))[0]
            elif effect == "SplitToning":
                result_img = MrWeazSplitToning().process(image, kwargs.get("highlight_color", "#FFCC00"), kwargs.get("shadow_color", "#003366"), kwargs.get("balance", 0.5))[0]
            elif effect == "CRTVHS":
                result_img = MrWeazCRTVHS().process(image, kwargs.get("scanline_intensity", 0.2), kwargs.get("rgb_shift_amount", 0.005), kwargs.get("noise_amount", 0.1), kwargs.get("aberration_strength", 0.0))[0]
            elif effect == "FilmGrainVignette":
                result_img = MrWeazFilmGrainVignette().process(image, kwargs.get("vignette_intensity", 0.3), kwargs.get("grain_amount", 0.1))[0]
            elif effect == "Posterize":
                result_img = MrWeazPosterize().process(image, kwargs.get("levels", 4))[0]
            elif effect == "Letterbox":
                result_img = MrWeazLetterbox().process(image, kwargs.get("bar_ratio", 0.1), kwargs.get("bar_color", "Black"), kwargs.get("orientation", "Horizontal (Top/Bottom)"))[0]
            elif effect == "ColorAdjust":
                result_img = MrWeazColorAdjust().process(image, kwargs.get("brightness", 1.0), kwargs.get("contrast", 1.0), kwargs.get("saturation", 1.0), kwargs.get("hue_shift", 0.0))[0]
            elif effect == "VibeTransfer":
                ref = kwargs.get("reference_image", image)
                result_img = MrWeazVibeTransfer().process(image, ref, kwargs.get("intensity", 1.0))[0]
            elif effect == "LensAberration":
                result_img = MrWeazLensAberration().process(image, kwargs.get("aberration_strength", 0.02))[0]
            elif effect == "Pixelate":
                result_img = MrWeazPixelate().process(image, kwargs.get("dot_size", 4))[0]
            elif effect == "Solarize":
                result_img = MrWeazSolarize().process(image, kwargs.get("threshold", 0.8))[0]
            elif effect == "Duotone":
                result_img = MrWeazDuotone().process(
                    image,
                    kwargs.get("highlight_color", "#FFCC00"),
                    kwargs.get("shadow_color", "#003366"),
                    kwargs.get("intensity", 1.0)
                )[0]
            elif effect == "Emboss":
                result_img = MrWeazEmboss().process(image, kwargs.get("strength", 1.0), kwargs.get("angle", 45.0))[0]
            elif effect == "VignetteOnly":
                result_img = MrWeazVignetteOnly().process(image, kwargs.get("vignette_intensity", 0.3))[0]
            elif effect == "Gamma":
                result_img = MrWeazGamma().process(image, kwargs.get("brightness", 1.0))[0]
            elif effect == "CrossProcess":
                result_img = MrWeazCrossProcess().process(image, kwargs.get("intensity", 1.0))[0]
            elif effect == "ChromaShift":
                result_img = MrWeazChromaShift().process(image, kwargs.get("rgb_shift_amount", 0.005))[0]
            elif effect == "Sketch":
                result_img = MrWeazSketch().process(image, kwargs.get("sensitivity", 1.0))[0]
            elif effect == "Vibrance":
                result_img = MrWeazVibrance().process(image, kwargs.get("saturation", 1.0), kwargs.get("intensity", 1.0))[0]
            elif effect == "NightVision":
                result_img = MrWeazNightVision().process(image, kwargs.get("intensity", 1.0), kwargs.get("noise_amount", 0.1))[0]

        # Optional mask blending: apply effect only where mask is white.
        mask = kwargs.get("mask", None)
        if mask is not None:
            mask_prepared = self._prepare_mask(mask, image)
            result_img = torch.lerp(image, result_img, mask_prepared)

        # Save both for UI comparison
        saved_before = self.save_images(image, filename_prefix="mrweaz.aio.before", prompt=prompt, extra_pnginfo=extra_pnginfo)
        saved_after = self.save_images(result_img, filename_prefix="mrweaz.aio.after", prompt=prompt, extra_pnginfo=extra_pnginfo)

        return {
            "ui": {
                "before": saved_before["ui"]["images"],
                "after": saved_after["ui"]["images"]
            },
            "result": (result_img,)
        }

# ──────────────────────────────────────────────────────────────
# NEW BASIC EFFECTS
# ──────────────────────────────────────────────────────────────

class MrWeazGaussianBlur:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image": ("IMAGE",),
            "radius": ("INT", {"default": 4, "min": 1, "max": 128, "step": 1}),
            "sigma": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
        }}
    RETURN_TYPES = ("IMAGE",); FUNCTION = "process"; CATEGORY = "MrWeazNodes/Effects"
    def process(self, image, radius, sigma):
        img = image.permute(0, 3, 1, 2)
        kernel_size = radius * 2 + 1
        x = torch.arange(kernel_size).float() - radius
        kernel_1d = torch.exp(-0.5 * (x / sigma)**2)
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = kernel_1d.view(1, 1, 1, -1) * kernel_1d.view(1, 1, -1, 1)
        kernel_2d = kernel_2d.repeat(3, 1, 1, 1).to(img.device)
        padding = radius
        img = F.pad(img, (padding, padding, padding, padding), mode='reflect')
        img = F.conv2d(img, kernel_2d, groups=3)
        return (img.permute(0, 2, 3, 1),)

class MrWeazSharpen:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image": ("IMAGE",),
            "amount": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.05}),
            "radius": ("INT", {"default": 1, "min": 1, "max": 5}),
        }}
    RETURN_TYPES = ("IMAGE",); FUNCTION = "process"; CATEGORY = "MrWeazNodes/Effects"
    def process(self, image, amount, radius):
        img = image.permute(0, 3, 1, 2)
        # Simple laplacian-based sharpen
        kernel = torch.tensor([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]).float().view(1, 1, 3, 3)
        if radius > 1: # Basic scaling of the kernel effect
             kernel = F.interpolate(kernel, size=(radius*2+1, radius*2+1), mode='bilinear')
        kernel = kernel.repeat(3, 1, 1, 1).to(img.device)
        padding = kernel.shape[2] // 2
        blurred = F.conv2d(F.pad(img, (padding, padding, padding, padding), mode='reflect'), kernel, groups=3)
        result = torch.lerp(img, blurred, amount)
        return (torch.clamp(result, 0, 1).permute(0, 2, 3, 1),)

class MrWeazInvert:
    @classmethod
    def INPUT_TYPES(s): return {"required": {"image": ("IMAGE",)}}
    RETURN_TYPES = ("IMAGE",); FUNCTION = "process"; CATEGORY = "MrWeazNodes/Effects"
    def process(self, image): return (1.0 - image,)

class MrWeazSepia:
    @classmethod
    def INPUT_TYPES(s): return {"required": {"image": ("IMAGE",), "intensity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05})}}
    RETURN_TYPES = ("IMAGE",); FUNCTION = "process"; CATEGORY = "MrWeazNodes/Effects"
    def process(self, image, intensity):
        sepia_matrix = torch.tensor([
            [0.393, 0.769, 0.189],
            [0.349, 0.686, 0.168],
            [0.272, 0.534, 0.131]
        ]).to(image.device)
        res = image @ sepia_matrix.T
        return (torch.lerp(image, torch.clamp(res, 0, 1), intensity),)

class MrWeazEdgeDetection:
    @classmethod
    def INPUT_TYPES(s): return {"required": {"image": ("IMAGE",), "sensitivity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1})}}
    RETURN_TYPES = ("IMAGE",); FUNCTION = "process"; CATEGORY = "MrWeazNodes/Effects"
    def process(self, image, sensitivity):
        img = image.permute(0, 3, 1, 2).mean(dim=1, keepdim=True)
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).float().view(1, 1, 3, 3).to(img.device)
        ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).float().view(1, 1, 3, 3).to(img.device)
        gx = F.conv2d(F.pad(img, (1,1,1,1), mode='reflect'), kx)
        gy = F.conv2d(F.pad(img, (1,1,1,1), mode='reflect'), ky)
        edge = torch.sqrt(gx**2 + gy**2) * sensitivity
        return (torch.clamp(edge.repeat(1, 3, 1, 1), 0, 1).permute(0, 2, 3, 1),)

# ──────────────────────────────────────────────────────────────
# NEW ADVANCED EFFECTS
# ──────────────────────────────────────────────────────────────

class MrWeazGlitch:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image": ("IMAGE",),
            "glitch_amount": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.05}),
            "scan_distortion": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
        }}
    RETURN_TYPES = ("IMAGE",); FUNCTION = "process"; CATEGORY = "MrWeazNodes/Effects"
    def process(self, image, glitch_amount, scan_distortion, seed):
        torch.manual_seed(seed)
        b, h, w, c = image.shape
        img = image.clone()
        # Random horizontal slice shifts
        if glitch_amount > 0:
            num_slices = int(10 * glitch_amount)
            for _ in range(num_slices):
                slice_h = torch.randint(2, h // 4, (1,)).item()
                y_pos = torch.randint(0, h - slice_h, (1,)).item()
                shift = torch.randint(-int(w * 0.1 * glitch_amount), int(w * 0.1 * glitch_amount), (1,)).item()
                img[:, y_pos:y_pos+slice_h, :, :] = torch.roll(img[:, y_pos:y_pos+slice_h, :, :], shifts=shift, dims=2)
        
        # Color channel random offset
        if scan_distortion > 0:
            for i in [0, 2]: # R and B
                shift_x = torch.randint(-int(w * 0.02 * scan_distortion), int(w * 0.02 * scan_distortion), (1,)).item()
                shift_y = torch.randint(-int(h * 0.02 * scan_distortion), int(h * 0.02 * scan_distortion), (1,)).item()
                img[:, :, :, i] = torch.roll(img[:, :, :, i], shifts=(shift_y, shift_x), dims=(1, 2))
        return (torch.clamp(img, 0, 1),)

class MrWeazTiltShift:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image": ("IMAGE",),
            "blur_amount": ("INT", {"default": 16, "min": 0, "max": 64}),
            "focus_center": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
            "focus_width": ("FLOAT", {"default": 0.2, "min": 0.01, "max": 1.0, "step": 0.05}),
        }}
    RETURN_TYPES = ("IMAGE",); FUNCTION = "process"; CATEGORY = "MrWeazNodes/Effects"
    def process(self, image, blur_amount, focus_center, focus_width):
        if blur_amount <= 0: return (image,)
        # Create blurred version
        img_p = image.permute(0, 3, 1, 2)
        k = blur_amount * 2 + 1
        blurred = F.avg_pool2d(F.pad(img_p, (blur_amount, blur_amount, blur_amount, blur_amount), mode='reflect'), k, stride=1)
        # Gradient mask
        h = image.shape[1]
        # img_p is B,C,H,W, so mask must be broadcastable as 1,1,H,1.
        y = torch.linspace(0, 1, h, device=image.device).view(1, 1, h, 1)
        mask = torch.abs(y - focus_center) - (focus_width / 2)
        mask = torch.clamp(mask / (focus_width / 2), 0, 1) # 0 at focus, 1 at edges
        result = torch.lerp(img_p, blurred, mask)
        return (result.permute(0, 2, 3, 1),)

class MrWeazHalftone:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image": ("IMAGE",),
            "dot_size": ("INT", {"default": 4, "min": 1, "max": 24}),
            "angle": ("FLOAT", {"default": 45.0, "min": 0, "max": 360}),
            "monochrome": (["True", "False"],),
        }}
    RETURN_TYPES = ("IMAGE",); FUNCTION = "process"; CATEGORY = "MrWeazNodes/Effects"
    def process(self, image, dot_size, angle, monochrome):
        b, h, w, c = image.shape
        img = image.clone()
        if monochrome == "True":
            img = img.mean(dim=-1, keepdim=True).repeat(1, 1, 1, 3)
        yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        yy, xx = yy.to(image.device).float(), xx.to(image.device).float()
        rad = angle * 3.14159 / 180.0
        # Rotate coordinates for halftone angle
        cos_rad = math.cos(rad)
        sin_rad = math.sin(rad)
        rot_x = xx * cos_rad - yy * sin_rad
        rot_y = xx * sin_rad + yy * cos_rad
        # Sinusoidal pattern
        pattern = (torch.sin(rot_x / dot_size * 3.14159) + torch.sin(rot_y / dot_size * 3.14159)) / 2.0
        # The dot pattern is compared against the image intensity
        result = (img > (pattern.unsqueeze(0).unsqueeze(-1) + 0.5)).float()
        return (result,)

class MrWeazBloom:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image": ("IMAGE",),
            "threshold": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05}),
            "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 5.0, "step": 0.1}),
            "blur_radius": ("INT", {"default": 24, "min": 1, "max": 128}),
        }}
    RETURN_TYPES = ("IMAGE",); FUNCTION = "process"; CATEGORY = "MrWeazNodes/Effects"
    def process(self, image, threshold, strength, blur_radius):
        # Extract bright areas
        bright = torch.clamp(image - threshold, 0, 1) / (1.0 - threshold + 1e-5)
        # Blur the bright areas
        bright_p = bright.permute(0, 3, 1, 2)
        k = blur_radius * 2 + 1
        blurred = F.avg_pool2d(F.pad(bright_p, (blur_radius, blur_radius, blur_radius, blur_radius), mode='reflect'), k, stride=1)
        # Add back to original
        bloom = blurred.permute(0, 2, 3, 1) * strength
        result = image + bloom
        return (torch.clamp(result, 0, 1),)

class MrWeazSplitToning:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image": ("IMAGE",),
            "highlight_color": ("COLOR", {"default": "#FFCC00"}),
            "shadow_color": ("COLOR", {"default": "#003366"}),
            "balance": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
        }}
    RETURN_TYPES = ("IMAGE",); FUNCTION = "process"; CATEGORY = "MrWeazNodes/Effects"
    def process(self, image, highlight_color, shadow_color, balance):
        def hex_to_rgb(h):
            h = h.lstrip('#'); return torch.tensor([int(h[i:i+2], 16)/255.0 for i in (0, 2, 4)]).to(image.device)
        h_rgb = hex_to_rgb(highlight_color).view(1, 1, 1, 3)
        s_rgb = hex_to_rgb(shadow_color).view(1, 1, 1, 3)
        luma = (0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]).unsqueeze(-1)
        # Split point based on balance
        mask = torch.clamp((luma - (1.0 - balance)) / max(0.01, balance), 0, 1)
        toned_h = image * (1.0 - mask) + (image * h_rgb) * mask
        toned_s = image * mask + (image * s_rgb) * (1.0 - mask)
        result = torch.lerp(toned_s, toned_h, mask)
        return (torch.clamp(result, 0, 1),)


class MrWeazPixelate:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image": ("IMAGE",),
            "block_size": ("INT", {"default": 4, "min": 1, "max": 64}),
        }}
    RETURN_TYPES = ("IMAGE",); FUNCTION = "process"; CATEGORY = "MrWeazNodes/Effects"
    def process(self, image, block_size):
        if block_size <= 1:
            return (image,)
        img = image.permute(0, 3, 1, 2)
        _, _, h, w = img.shape
        small_h = max(1, h // block_size)
        small_w = max(1, w // block_size)
        down = F.interpolate(img, size=(small_h, small_w), mode='nearest')
        up = F.interpolate(down, size=(h, w), mode='nearest')
        return (up.permute(0, 2, 3, 1),)


class MrWeazSolarize:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image": ("IMAGE",),
            "threshold": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05}),
        }}
    RETURN_TYPES = ("IMAGE",); FUNCTION = "process"; CATEGORY = "MrWeazNodes/Effects"
    def process(self, image, threshold):
        t = float(torch.clamp(torch.tensor(threshold, device=image.device), 0.0, 1.0))
        result = torch.where(image < t, image, 1.0 - image)
        return (torch.clamp(result, 0.0, 1.0),)


class MrWeazDuotone:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image": ("IMAGE",),
            "highlight_color": ("COLOR", {"default": "#FFCC00"}),
            "shadow_color": ("COLOR", {"default": "#003366"}),
            "intensity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
        }}
    RETURN_TYPES = ("IMAGE",); FUNCTION = "process"; CATEGORY = "MrWeazNodes/Effects"
    def process(self, image, highlight_color, shadow_color, intensity):
        def hex_to_rgb(h):
            h = h.lstrip('#')
            return torch.tensor([int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4)], device=image.device).view(1, 1, 1, 3)
        hi = hex_to_rgb(highlight_color)
        sh = hex_to_rgb(shadow_color)
        luma = (0.299 * image[..., 0:1] + 0.587 * image[..., 1:2] + 0.114 * image[..., 2:3])
        toned = sh + luma * (hi - sh)
        result = torch.lerp(image, toned, intensity)
        return (torch.clamp(result, 0.0, 1.0),)


class MrWeazEmboss:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image": ("IMAGE",),
            "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05}),
            "angle": ("FLOAT", {"default": 45.0, "min": 0.0, "max": 360.0, "step": 1.0}),
        }}
    RETURN_TYPES = ("IMAGE",); FUNCTION = "process"; CATEGORY = "MrWeazNodes/Effects"
    def process(self, image, strength, angle):
        img = image.permute(0, 3, 1, 2)
        # Directional gradient-based emboss from chosen angle.
        rad = angle * (math.pi / 180.0)
        dx = int(round(math.cos(rad)))
        dy = int(round(math.sin(rad)))
        shifted = torch.roll(img, shifts=(dy, dx), dims=(2, 3))
        grad = img - shifted
        embossed = torch.clamp(0.5 + grad * strength, 0.0, 1.0)
        result = torch.lerp(img, embossed, min(max(strength / 2.0, 0.0), 1.0))
        return (torch.clamp(result, 0.0, 1.0).permute(0, 2, 3, 1),)


class MrWeazVignetteOnly:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image": ("IMAGE",),
            "vignette_intensity": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),
        }}
    RETURN_TYPES = ("IMAGE",); FUNCTION = "process"; CATEGORY = "MrWeazNodes/Effects"
    def process(self, image, vignette_intensity):
        if vignette_intensity <= 0.0:
            return (image,)
        b, h, w, c = image.shape
        y = torch.linspace(-1, 1, h, device=image.device).view(1, h, 1, 1)
        x = torch.linspace(-1, 1, w, device=image.device).view(1, 1, w, 1)
        radius = torch.sqrt(x**2 + y**2)
        vignette = 1.0 - torch.clamp(radius * vignette_intensity, 0.0, 1.0)
        return (torch.clamp(image * vignette, 0.0, 1.0),)


class MrWeazGamma:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image": ("IMAGE",),
            "brightness": ("FLOAT", {"default": 1.0, "min": 0.2, "max": 3.0, "step": 0.05}),
        }}
    RETURN_TYPES = ("IMAGE",); FUNCTION = "process"; CATEGORY = "MrWeazNodes/Effects"
    def process(self, image, brightness):
        g = max(0.001, float(brightness))
        corrected = torch.pow(torch.clamp(image, 0.0, 1.0), 1.0 / g)
        return (torch.clamp(corrected, 0.0, 1.0),)


class MrWeazCrossProcess:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image": ("IMAGE",),
            "intensity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
        }}
    RETURN_TYPES = ("IMAGE",); FUNCTION = "process"; CATEGORY = "MrWeazNodes/Effects"
    def process(self, image, intensity):
        i = float(min(max(intensity, 0.0), 1.0))
        # Classic-ish cross-process curve: warmer highs, cooler shadows, contrast bump.
        r = torch.clamp(image[..., 0:1] * 1.08 + 0.02, 0.0, 1.0)
        g = torch.clamp(torch.pow(image[..., 1:2], 0.95), 0.0, 1.0)
        b = torch.clamp(torch.pow(image[..., 2:3], 1.08) * 0.95, 0.0, 1.0)
        xp = torch.cat([r, g, b], dim=-1)
        result = torch.lerp(image, xp, i)
        return (torch.clamp(result, 0.0, 1.0),)


class MrWeazChromaShift:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image": ("IMAGE",),
            "rgb_shift_amount": ("FLOAT", {"default": 0.005, "min": 0.0, "max": 0.1, "step": 0.001}),
        }}
    RETURN_TYPES = ("IMAGE",); FUNCTION = "process"; CATEGORY = "MrWeazNodes/Effects"
    def process(self, image, rgb_shift_amount):
        if rgb_shift_amount <= 0.0:
            return (image,)
        b, h, w, c = image.shape
        shift = max(1, int(w * rgb_shift_amount))
        out = image.clone()
        out[..., 0] = torch.roll(image[..., 0], shifts=-shift, dims=2)
        out[..., 2] = torch.roll(image[..., 2], shifts=shift, dims=2)
        return (torch.clamp(out, 0.0, 1.0),)


class MrWeazSketch:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image": ("IMAGE",),
            "sensitivity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1}),
        }}
    RETURN_TYPES = ("IMAGE",); FUNCTION = "process"; CATEGORY = "MrWeazNodes/Effects"
    def process(self, image, sensitivity):
        img = image.permute(0, 3, 1, 2)
        gray = img.mean(dim=1, keepdim=True)
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=image.device).float().view(1, 1, 3, 3)
        ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=image.device).float().view(1, 1, 3, 3)
        gx = F.conv2d(F.pad(gray, (1, 1, 1, 1), mode='reflect'), kx)
        gy = F.conv2d(F.pad(gray, (1, 1, 1, 1), mode='reflect'), ky)
        edge = torch.sqrt(gx**2 + gy**2) * sensitivity
        sketch = 1.0 - torch.clamp(edge, 0.0, 1.0)
        sketch_rgb = sketch.repeat(1, 3, 1, 1).permute(0, 2, 3, 1)
        return (torch.clamp(sketch_rgb, 0.0, 1.0),)


class MrWeazVibrance:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image": ("IMAGE",),
            "saturation": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05}),
            "intensity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
        }}
    RETURN_TYPES = ("IMAGE",); FUNCTION = "process"; CATEGORY = "MrWeazNodes/Effects"
    def process(self, image, saturation, intensity):
        i = float(min(max(intensity, 0.0), 1.0))
        sat = float(max(0.0, saturation))
        luma = (0.299 * image[..., 0:1] + 0.587 * image[..., 1:2] + 0.114 * image[..., 2:3])
        color_delta = image - luma
        # Push more color in flatter areas, preserve vivid highlights.
        adaptive = 1.0 + (sat - 1.0) * (1.0 - torch.abs(color_delta).mean(dim=-1, keepdim=True))
        boosted = torch.clamp(luma + color_delta * adaptive, 0.0, 1.0)
        result = torch.lerp(image, boosted, i)
        return (torch.clamp(result, 0.0, 1.0),)


class MrWeazNightVision:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image": ("IMAGE",),
            "intensity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            "noise_amount": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.05}),
        }}
    RETURN_TYPES = ("IMAGE",); FUNCTION = "process"; CATEGORY = "MrWeazNodes/Effects"
    def process(self, image, intensity, noise_amount):
        i = float(min(max(intensity, 0.0), 1.0))
        luma = (0.299 * image[..., 0:1] + 0.587 * image[..., 1:2] + 0.114 * image[..., 2:3])
        green_tint = torch.cat([luma * 0.2, luma * 1.1, luma * 0.2], dim=-1)
        if noise_amount > 0:
            noise = (torch.rand_like(green_tint) - 0.5) * noise_amount * 0.25
            green_tint = green_tint + noise
        result = torch.lerp(image, green_tint, i)
        return (torch.clamp(result, 0.0, 1.0),)

NODE_CLASS_MAPPINGS = {
    "MrWeazAIOImageEffects": MrWeazAIOImageEffects
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MrWeazAIOImageEffects": "(MRW) All-in-One Image Effects"
}
