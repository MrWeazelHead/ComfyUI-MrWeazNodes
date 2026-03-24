# ComfyUI-MrWeazNodes

A collection of professional, high-quality custom nodes for ComfyUI, designed by MrWeazel. This pack focuses on enhanced UI experiences, creative image post-processing, and streamlined Flux (Klein) editing workflows.

---

## 🚀 Featured Nodes

### 🎨 (MRW) Advanced Prompt Studio
A state-of-the-art prompt management interface directly inside ComfyUI.
- **Symmetrical UI**: Color-coded Positive (Green) and Negative (Red) prompt areas.
- **Presets & Logging**: Save your favorite prompts and view historically used ones with the integrated logger.
- **Dynamic Data**: Automatically manages local storage for presets in the `data/` folder.
- **Slick Styling**: Dark-mode optimized with tinted backgrounds for better visual focus.

### ⚖️ (MRW) Pro Image Comparer
A high-performance comparison tool for side-by-side or layered evaluation.
- **Slide Mode**: Interactive vertical slider reveals Image B as you hover over the node.
- **Click Mode**: Hold the mouse button down to instantly toggle between Image A and B.
- **Pro Metadata**: Saves both comparison images with full metadata support for future reference.
- **Custom Context Menu**: Easily switch modes or toggle informational labels via right-click.

### 🪄 (MRW) All-in-One Image Effects
A unified post-processing powerhouse that consolidates dozens of effects into a single node.
- **Dynamic UI**: Sizable node that only shows parameters relevant to the selected effect.
- **Effects Library**: Includes Gaussian Blur, Sharpen, Sepia, Edge Detection, CRT/VHS (with radial aberration), Halftone, Bloom, Tilt-Shift, Glitch, and many more.
- **Performance Optimized**: Built using efficient PyTorch operations for fast inference.

---

## ⚡ Flux Klein QoL Suite
Specially designed nodes for the Flux generation ecosystem.
- **(MRW) Klein Quick Edit Ref Loader**: Encodes reference images for precise editing workflows.
- **(MRW) Prompt Blender**: A dedicated string-building node for organizing subject, action, environment, and style components.
- **(MRW) Reference Grid Stitcher**: Quickly stitch and arrange reference images horizontally or vertically for comparison.
- **(MRW) Mask Feather & Expand**: Professional-grade mask post-processing for smooth inpainting transitions.
- **(MRW) Batch Seed Runner**: Deterministic seed generator for running varied batch iterations.

---

## 📦 The "Extras"

- **(MRW) Resolution Selector**: A massive collection of preset resolutions for Web, Social Media, Devices, Print, and Cinema. Supports 4-channel and 16-channel (SD3.5) scaling.
- **(MRW) Pixel Art Downscale**: Turn any image into high-quality pixel art with dithering, quantization, and retro color schemes (e.g., Gameboy).
- **(MRW) All-in-One Hires Fix KSampler**: A consolidated sampling node that handles base generation and hires-fix upscaling in a single step with quality presets.

---

## 🛠️ Installation

1. Navigate to your ComfyUI `custom_nodes` folder.
2. Clone this repository:
   ```bash
   git clone https://github.com/MrWeazelHead/ComfyUI-MrWeazNodes.git
   ```
3. Restart ComfyUI.

## 📝 Tips
- **Hard Refresh**: If you don't see the new UI elements for the Prompt Studio or Image Comparer, perform a hard refresh (**Ctrl + F5**) in your browser.
- **Right-Click**: Many nodes have hidden features in their right-click context menu (e.g., switching modes on the Image Comparer).

---

Developed with ❤️ by **MrWeazel**.
