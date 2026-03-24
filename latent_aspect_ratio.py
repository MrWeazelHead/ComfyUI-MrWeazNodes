#MrWeaz Nodes - (MRW) Resolution Selector

import torch

RESOLUTIONS = {
    # Custom
    "Custom - Manual Override": (0, 0),
    
    # Web / Screen (Desktop & Ultrawide)
    "Web - 720p HD (1280 x 720)": (1280, 720),
    "Web - 1080p FHD (1920 x 1080)": (1920, 1080),
    "Web - 1440p QHD (2560 x 1440)": (2560, 1440),
    "Web - 4K UHD (3840 x 2160)": (3840, 2160),
    "Web - 5K (5120 x 2880)": (5120, 2880),
    "Web - 8K UHD (7680 x 4320)": (7680, 4320),
    "Web - Ultrawide 1080p (2560 x 1080)": (2560, 1080),
    "Web - Ultrawide 1440p (3440 x 1440)": (3440, 1440),
    "Web - Super Ultrawide (5120 x 1440)": (5120, 1440),
    "Web - Ultrawide 4K (5120 x 2160)": (5120, 2160),
    
    # Devices (Apple / Android approximations)
    "Device - iPhone 14/15 Pro (1176 x 2552)": (1176, 2552),
    "Device - iPhone 14/15 Pro Max (1288 x 2792)": (1288, 2792),
    "Device - iPad Pro 11-inch (1664 x 2384)": (1664, 2384),
    "Device - iPad Pro 12.9-inch (2048 x 2736)": (2048, 2736),
    "Device - Android 1080p Phone (1080 x 2400)": (1080, 2400),
    "Device - Android 1440p Phone (1440 x 3200)": (1440, 3200),
    
    # Social Media
    "Social - IG Square (1080 x 1080)": (1080, 1080),
    "Social - IG Portrait (1080 x 1352)": (1080, 1352),
    "Social - IG Story/Reel (1080 x 1920)": (1080, 1920),
    "Social - Twitter/FB Feed (1280 x 720)": (1280, 720),
    "Social - Twitter Header (1504 x 504)": (1504, 504),
    "Social - FB Cover (824 x 312)": (824, 312),
    "Social - YouTube Thumbnail (1280 x 720)": (1280, 720),
    "Social - YouTube Channel Art (2560 x 1440)": (2560, 1440),
    "Social - LinkedIn Banner (1584 x 392)": (1584, 392),
    "Social - TikTok Post (1080 x 1920)": (1080, 1920),
    "Social - Twitch Banner (1920 x 480)": (1920, 480),
    
    # Print (300 DPI, adjusted to multiples of 8)
    "Print - A6 (1240 x 1752)": (1240, 1752),
    "Print - A5 (1744 x 2480)": (1744, 2480),
    "Print - A4 (2480 x 3504)": (2480, 3504),
    "Print - A3 (3504 x 4960)": (3504, 4960),
    "Print - A2 (4960 x 7016)": (4960, 7016),
    "Print - US Letter (2552 x 3304)": (2552, 3304),
    "Print - US Legal (2552 x 4200)": (2552, 4200),
    "Print - Tabloid (3304 x 5104)": (3304, 5104),
    "Print - Business Card (1048 x 600)": (1048, 600),
    
    # Magazine & Editorial (300 DPI, adjusted to multiples of 8)
    "Magazine - Standard Cover (2512 x 3264)": (2512, 3264),
    "Magazine - Digest Size (1648 x 2480)": (1648, 2480),
    "Magazine - Fashion Spread (4960 x 3304)": (4960, 3304),
    "Magazine - Two-Page Spread (5024 x 3264)": (5024, 3264),
    "Comic Book - Standard Page (1984 x 3056)": (1984, 3056),
    "Comic Book - Manga B6 (1512 x 2152)": (1512, 2152),
    
    # Photography & Art (Common Framing Ratios)
    "Photo - 4x6 Postcard (1200 x 1800)": (1200, 1800),
    "Photo - 5x7 Standard (1504 x 2104)": (1504, 2104),
    "Photo - 8x10 Portrait (2400 x 3000)": (2400, 3000),
    "Photo - 11x14 Fine Art (3304 x 4200)": (3304, 4200),
    "Photo - Medium Format 6x4.5 (1344 x 1008)": (1344, 1008),
    "Photo - Medium Format 6x6 (2048 x 2048)": (2048, 2048),
    "Photo - Medium Format 6x7 (2104 x 1800)": (2104, 1800),
    "Photo - Large Format 4x5 (2400 x 3000)": (2400, 3000),
    "Photo - Panoramic 2:1 (3008 x 1504)": (3008, 1504),
    "Photo - Panoramic 3:1 (4512 x 1504)": (4512, 1504),
    
    # Film / Cinema
    "Cinema - 1080p (1920 x 1080)": (1920, 1080),
    "Cinema - 2K Flat 1.85:1 (2000 x 1080)": (2000, 1080),
    "Cinema - 2K Scope 2.39:1 (2048 x 856)": (2048, 856),
    "Cinema - 4K Flat 1.85:1 (4000 x 2160)": (4000, 2160),
    "Cinema - 4K Scope 2.39:1 (4096 x 1720)": (4096, 1720),
    "Cinema - 8K UHD (7680 x 4320)": (7680, 4320),
    "16mm Film - Standard 1.37:1 (1024 x 744)": (1024, 744),
    "35mm Film - Standard (3200 x 2400)": (3200, 2400),
    "35mm Film - Aspect 3:2 (3600 x 2400)": (3600, 2400),
    "70mm Film - Todd-AO 2.20:1 (4096 x 1864)": (4096, 1864),
    "70mm Film - IMAX (5856 x 4096)": (5856, 4096),
    
    # Basic Generation Base Ratios (Common MP values)
    "Basic - 1:1 Square (1024 x 1024)": (1024, 1024),
    "Basic - 4:3 Photography Standard (1152 x 896)": (1152, 896),
    "Basic - 3:4 Photography Portrait (896 x 1152)": (896, 1152),
    "Basic - 3:2 Classic Photography (1216 x 832)": (1216, 832),
    "Basic - 2:3 Classic Portrait (832 x 1216)": (832, 1216),
    "Basic - 16:9 Widescreen (1344 x 768)": (1344, 768),
    "Basic - 9:16 Mobile Portrait (768 x 1344)": (768, 1344),
    "Basic - 21:9 Cinematic Widescreen (1536 x 640)": (1536, 640),
}

class MrWeazLatentAspectRatio:
    LATENT_FORMATS = {
        "SD 1.5 / SDXL (4 channel)": {"channels": 4, "downscale": 8},
        "SD 3.5 (16 channel)":        {"channels": 16, "downscale": 8},
        "Flux 2 (128 channel)":       {"channels": 128, "downscale": 16},
    }

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "resolution": (list(RESOLUTIONS.keys()), {"default": "Basic - 1:1 Square (1024 x 1024)"}),
                "latent_type": (list(s.LATENT_FORMATS.keys()), {"default": "Flux 2 (128 channel)"}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
                "use_custom_override": (["False", "True"], {"default": "False"}),
                "custom_width": ("INT", {"default": 1024, "min": 64, "max": 16384, "step": 8}),
                "custom_height": ("INT", {"default": 1024, "min": 64, "max": 16384, "step": 8}),
                "swap_dimensions": (["Off", "On"],),
            }
        }
    
    RETURN_TYPES = ("LATENT", "INT", "INT")
    RETURN_NAMES = ("latent", "width", "height")
    FUNCTION = "generate"
    CATEGORY = "MrWeazNodes/Latents"

    def generate(self, resolution, latent_type, batch_size, use_custom_override, custom_width, custom_height, swap_dimensions):
        if use_custom_override == "True":
            width = custom_width
            height = custom_height
        else:
            width, height = RESOLUTIONS.get(resolution, (1024, 1024))
            
        if swap_dimensions == "On":
            width, height = height, width
            
        # Ensure perfectly safe sizes (multiples of 8)
        width = int((width // 8) * 8)
        height = int((height // 8) * 8)

        # Build the empty latent
        fmt = self.LATENT_FORMATS[latent_type]
        channels = fmt["channels"]
        downscale = fmt["downscale"]
        latent_height = height // downscale
        latent_width = width // downscale
        latent = torch.zeros([batch_size, channels, latent_height, latent_width])
            
        return ({"samples": latent}, width, height)

NODE_CLASS_MAPPINGS = {
    "MrWeazLatentAspectRatio": MrWeazLatentAspectRatio
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MrWeazLatentAspectRatio": "(MRW) Resolution Selector"
}
