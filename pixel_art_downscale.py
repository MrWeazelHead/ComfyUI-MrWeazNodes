#MrWeaz Nodes - (MRW) Pixel Art Downscaler

import torch
import torch.nn.functional as F

class MrWeazPixelArtDownscale:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "downscale_factor": ("INT", {"default": 4, "min": 1, "max": 128, "step": 1}),
                "downscale_mode": (["nearest", "area", "bilinear", "bicubic"], {"default": "nearest"}),
                "upscale_back": (["True", "False"], {"default": "True"}),
                "color_palette_levels": ("INT", {"default": 0, "min": 0, "max": 256, "step": 1, "tooltip": "Used only for RGB Quantize and Grayscale. 0 to disable."}),
                "color_scheme": (["RGB Quantize", "Grayscale", "Gameboy", "Black & White", "None"], {"default": "None"}),
                "dither_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("IMAGE", "width", "height")
    FUNCTION = "process"
    CATEGORY = "MrWeazNodes/Image"

    def process(self, image, downscale_factor, downscale_mode, upscale_back, color_palette_levels, color_scheme, dither_strength):
        batch_size, height, width, channels = image.shape
        
        # permute to [B, C, H, W]
        img_permuted = image.permute(0, 3, 1, 2)
        
        if downscale_factor > 1:
            new_height = max(1, height // downscale_factor)
            new_width = max(1, width // downscale_factor)
            
            # Area interpolation is generally better for downscaling pixels nicely before nearest upscaling
            downscaled = F.interpolate(img_permuted, size=(new_height, new_width), mode=downscale_mode)
        else:
            downscaled = img_permuted
            new_height = height
            new_width = width
            
        if color_scheme != "None" or dither_strength > 0:
            # Apply dithering noise if requested to simulate pixel art dithering
            if dither_strength > 0.0:
                noise = (torch.rand_like(downscaled) - 0.5) * dither_strength
                downscaled = torch.clamp(downscaled + noise, 0.0, 1.0)
                
            if color_scheme == "Grayscale":
                gray = 0.299 * downscaled[:, 0:1, :, :] + 0.587 * downscaled[:, 1:2, :, :] + 0.114 * downscaled[:, 2:3, :, :]
                downscaled = gray.repeat(1, 3, 1, 1)
                
            elif color_scheme == "Gameboy":
                gray = 0.299 * downscaled[:, 0:1, :, :] + 0.587 * downscaled[:, 1:2, :, :] + 0.114 * downscaled[:, 2:3, :, :]
                # Quantize to exactly 4 distinct shades of retro green
                quantized = torch.round(gray * 3) / 3.0
                gameboy_c1 = torch.tensor([0.05, 0.21, 0.05]).view(1, 3, 1, 1).to(downscaled.device)
                gameboy_c4 = torch.tensor([0.60, 0.73, 0.05]).view(1, 3, 1, 1).to(downscaled.device)
                downscaled = gameboy_c1 + quantized * (gameboy_c4 - gameboy_c1)
                
            elif color_scheme == "Black & White":
                gray = 0.299 * downscaled[:, 0:1, :, :] + 0.587 * downscaled[:, 1:2, :, :] + 0.114 * downscaled[:, 2:3, :, :]
                bw = torch.round(gray)
                downscaled = bw.repeat(1, 3, 1, 1)
                
            # If standard RGB or Grayscale, check if we need to quantize the levels
            if (color_scheme == "RGB Quantize" or color_scheme == "Grayscale") and color_palette_levels > 0:
                levels = max(2, color_palette_levels)
                downscaled = torch.round(downscaled * (levels - 1)) / (levels - 1)
        
        # Clamp to ensure image valid format
        downscaled = torch.clamp(downscaled, 0.0, 1.0)
        
        if upscale_back == "True" and downscale_factor > 1:
            result_permuted = F.interpolate(downscaled, size=(height, width), mode="nearest")
            final_width = width
            final_height = height
        else:
            result_permuted = downscaled
            final_width = new_width
            final_height = new_height
            
        # permute back to [B, H, W, C]
        result = result_permuted.permute(0, 2, 3, 1)
        
        return (result, final_width, final_height)        

NODE_CLASS_MAPPINGS = {
    "MrWeazPixelArtDownscale": MrWeazPixelArtDownscale
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MrWeazPixelArtDownscale": "(MRW) Pixel Art Downscale"
}
