#MrWeaz Nodes - All In One KSampler High Res Fix

import torch
import torch.nn.functional as F
import nodes
import comfy.samplers

KLEIN_PRESETS = {
    "Quick (4-step distilled)":   {"steps": 4,  "cfg": 1.0, "sampler": "euler",  "scheduler": "simple"},
    "Quality (8-step distilled)": {"steps": 8,  "cfg": 1.0, "sampler": "euler",  "scheduler": "simple"},
    "Base Standard (20-step)":    {"steps": 20, "cfg": 3.5, "sampler": "euler",  "scheduler": "normal"},
    "Base High Quality (30-step)":{"steps": 30, "cfg": 3.5, "sampler": "euler",  "scheduler": "normal"},
}

class MrWeazHiresFixKSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "model": ("MODEL",),
                    "vae": ("VAE",),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "preset": (list(KLEIN_PRESETS.keys()), {"default": "Quality (8-step distilled)"}),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "latent_image": ("LATENT", ),
                    "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                    # Hires Fix
                    "enable_hires_fix": (["True", "False"], {"default": "True"}),
                    "upscale_method": (["nearest-exact", "bilinear", "area", "bicubic"],),
                    "upscale_factor": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 4.0, "step": 0.05}),
                    "hires_steps": ("INT", {"default": 6, "min": 1, "max": 10000}),
                    "hires_denoise": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.01}),
                    }
                }

    RETURN_TYPES = ("LATENT", "IMAGE", "LATENT", "IMAGE")
    RETURN_NAMES = ("base_latent", "base_image", "hires_latent", "hires_image")
    FUNCTION = "sample"
    CATEGORY = "MrWeazNodes/Core"

    def sample(self, model, vae, seed, preset, positive, negative, latent_image, denoise, enable_hires_fix, upscale_method, upscale_factor, hires_steps, hires_denoise, **kwargs):
        
        p = KLEIN_PRESETS[preset]
        steps = p["steps"]
        cfg = p["cfg"]
        sampler_name = p["sampler"]
        scheduler = p["scheduler"]

        # 1. Base Generation Pass
        base_latent = nodes.common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise)
        base_samples = base_latent[0]["samples"]
        base_image = vae.decode(base_samples)
        
        if enable_hires_fix == "False" or upscale_factor <= 1.0:
            return (base_latent[0], base_image, base_latent[0], base_image)
            
        # 2. Latent Upscale Pass
        b, c, h, w = base_samples.shape
        new_width = max(1, int(w * upscale_factor))
        new_height = max(1, int(h * upscale_factor))
        
        # interpolate
        upscaled_samples = F.interpolate(base_samples, size=(new_height, new_width), mode=upscale_method)
        upscaled_latent = {"samples": upscaled_samples}
        
        # 3. Hires Generation Pass
        hires_latent = nodes.common_ksampler(model, seed, hires_steps, cfg, sampler_name, scheduler, positive, negative, upscaled_latent, denoise=hires_denoise)
        hires_samples = hires_latent[0]["samples"]
        hires_image = vae.decode(hires_samples)
        
        return (base_latent[0], base_image, hires_latent[0], hires_image)

NODE_CLASS_MAPPINGS = {
    "MrWeazHiresFixKSampler": MrWeazHiresFixKSampler
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MrWeazHiresFixKSampler": "(MRW) All-in-One Hires Fix KSampler"
}
