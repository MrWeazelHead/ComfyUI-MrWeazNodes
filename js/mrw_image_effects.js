import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

/**
 * MrWeaz AIO Image Effects (Pro UI)
 * Features an integrated before/after comparison and styled controls.
 */

function imageDataToUrl(data) {
    if (!data) return "";
    return api.apiURL(`/view?filename=${encodeURIComponent(data.filename)}&type=${data.type}&subfolder=${data.subfolder}${app.getPreviewFormatParam()}${app.getRandParam()}`);
}

class MrWeazImageEffectsWidget {
    constructor(name, node) {
        this.name = name;
        this.node = node;
        this.type = "custom";
        this.value = { before: null, after: null };
        this.imgs = { before: new Image(), after: new Image() };
        this.loaded = { before: false, after: false };
        this.sliderX = 0.5;
        this.showComparison = true;
        
        // Parameter tracking
        this.activeParams = [];
    }

    setImages(before_data, after_data) {
        if (before_data) {
            this.loaded.before = false;
            this.imgs.before.onload = () => { this.loaded.before = true; this.node.setDirtyCanvas(true); };
            this.imgs.before.src = imageDataToUrl(before_data);
        }
        if (after_data) {
            this.loaded.after = false;
            this.imgs.after.onload = () => { this.loaded.after = true; this.node.setDirtyCanvas(true); };
            this.imgs.after.src = imageDataToUrl(after_data);
        }
    }

    draw(ctx, node, width, y, widget_offset) {
        const [nodeWidth, nodeHeight] = node.size;
        let currentY = y + 5; // Initial padding

        // 1. Comparison View (Top Section)
        if (this.showComparison) {
            const previewHeight = 220;
            const margin = 10;
            const radius = 8;
            
            // Background with subtle border
            ctx.fillStyle = "#0c0c0c";
            ctx.beginPath();
            ctx.roundRect(margin, currentY, width - margin * 2, previewHeight, radius);
            ctx.fill();
            ctx.strokeStyle = "#333";
            ctx.lineWidth = 1;
            ctx.stroke();

            const imgB = this.loaded.before ? this.imgs.before : null;
            const imgA = this.loaded.after ? this.imgs.after : null;

            if (imgB || imgA) {
                const master = imgB || imgA;
                const aspect = master.naturalWidth / master.naturalHeight;
                let targetW = width - margin * 3;
                let targetH = targetW / aspect;
                if (targetH > previewHeight - margin * 2) {
                    targetH = previewHeight - margin * 2;
                    targetW = targetH * aspect;
                }
                const offsetX = (width - targetW) / 2;
                const offsetY = currentY + (previewHeight - targetH) / 2;

                ctx.save();
                // Clip for rounded corners of the image itself
                ctx.beginPath();
                ctx.roundRect(offsetX, offsetY, targetW, targetH, 4);
                ctx.clip();
                
                if (imgB) ctx.drawImage(imgB, offsetX, offsetY, targetW, targetH);
                
                if (imgA) {
                    ctx.save();
                    ctx.beginPath();
                    const splitX = offsetX + (targetW * this.sliderX);
                    ctx.rect(splitX, offsetY, targetW - (splitX - offsetX), targetH);
                    ctx.clip();
                    ctx.drawImage(imgA, offsetX, offsetY, targetW, targetH);
                    ctx.restore();
                }
                ctx.restore();

                // Slider Line with Glow
                if (imgA && imgB) {
                    const splitX = offsetX + (targetW * this.sliderX);
                    
                    // Outer Glow
                    ctx.shadowBlur = 10;
                    ctx.shadowColor = "rgba(0, 255, 127, 0.5)";
                    ctx.strokeStyle = "#00ff7f";
                    ctx.lineWidth = 2;
                    ctx.setLineDash([8, 4]);
                    ctx.beginPath();
                    ctx.moveTo(splitX, offsetY);
                    ctx.lineTo(splitX, offsetY + targetH);
                    ctx.stroke();
                    ctx.setLineDash([]);
                    ctx.shadowBlur = 0;

                    // Slider Handle
                    ctx.fillStyle = "#00ff7f";
                    ctx.beginPath();
                    ctx.arc(splitX, offsetY + targetH / 2, 6, 0, Math.PI * 2);
                    ctx.fill();
                    ctx.strokeStyle = "#fff";
                    ctx.lineWidth = 1;
                    ctx.stroke();
                }
            } else {
                ctx.fillStyle = "#555";
                ctx.font = "italic 11px sans-serif";
                ctx.textAlign = "center";
                ctx.fillText("PREVIEW PENDING GENERATION", width/2, currentY + previewHeight/2);
            }
            currentY += previewHeight + 15;
        }

        // 2. Effect Header
        const effectWidget = this.node.widgets.find(w => w.name === "effect");
        const effectName = effectWidget?.value || "None";
        
        ctx.fillStyle = "#1e1e1e";
        ctx.beginPath();
        ctx.roundRect(10, currentY, width - 20, 34, 4);
        ctx.fill();
        ctx.strokeStyle = "#00ff7f";
        ctx.lineWidth = 0.5;
        ctx.stroke();
        
        ctx.fillStyle = "#00ff7f";
        ctx.font = "bold 13px Courier New";
        ctx.textAlign = "left";
        ctx.fillText(`🚀 [EFFECT]_${effectName.toUpperCase()}`, 25, currentY + 22);
    }

    computeSize(width) {
        return [width, this.showComparison ? 310 : 60];
    }
}

app.registerExtension({
    name: "MrWeaz.AIOImageEffectsProUI",
    async nodeCreated(node) {
        if (node.comfyClass === "MrWeazAIOImageEffects") {
            const effectWidgetsMap = {
                "None": [],
                "GaussianBlur": ["blur_radius", "sigma"],
                "Sharpen": ["strength", "blur_radius"],
                "Invert": [],
                "Sepia": ["intensity"],
                "EdgeDetection": ["sensitivity"],
                "Glitch": ["glitch_amount", "scan_distortion", "seed"],
                "TiltShift": ["blur_radius", "focus_center", "focus_width"],
                "Halftone": ["dot_size", "angle", "monochrome"],
                "Bloom": ["threshold", "strength", "blur_radius"],
                "SplitToning": ["highlight_color", "shadow_color", "balance"],
                "CRTVHS": ["scanline_intensity", "rgb_shift_amount", "noise_amount", "aberration_strength"],
                "FilmGrainVignette": ["vignette_intensity", "grain_amount"],
                "Posterize": ["levels"],
                "Letterbox": ["bar_ratio", "bar_color", "orientation"],
                "ColorAdjust": ["brightness", "contrast", "saturation", "hue_shift"],
                "VibeTransfer": ["intensity", "reference_image"],
                "LensAberration": ["aberration_strength"]
            };

            // 1. Add Custom Comparison Widget at the TOP
            const customUI = new MrWeazImageEffectsWidget("custom_ui", node);
            node.addCustomWidget(customUI);
            // Move to top
            const widgetIdx = node.widgets.indexOf(customUI);
            if (widgetIdx > -1) {
                node.widgets.splice(widgetIdx, 1);
                node.widgets.unshift(customUI);
            }
            node.customUI = customUI;

            node.getExtraMenuOptions = function(_, options) {
                options.push({
                    content: `${customUI.showComparison ? "Hide" : "Show"} Before/After Preview`,
                    callback: () => {
                        customUI.showComparison = !customUI.showComparison;
                        toggleWidgets();
                    }
                });
            };

            const toggleWidgets = () => {
                const effectWidget = node.widgets.find(w => w.name === "effect");
                if (!effectWidget) return;
                
                const selectedEffect = effectWidget.value;
                const visibleVars = effectWidgetsMap[selectedEffect] || [];
                
                // Show/Hide logic
                for (const w of node.widgets) {
                    if (w.name === "effect" || w.name === "image" || w.name === "custom_ui" || w.name === "reference_image") {
                        w.hidden = false;
                        if (w.type === "hidden" && w.name !== "custom_ui") w.type = w._mrwOldType || "number"; 
                        continue;
                    }
                    
                    const shouldShow = visibleVars.includes(w.name);
                    if (shouldShow) {
                        w.hidden = false;
                        if (w.type === "hidden") {
                            w.type = w._mrwOldType || "number"; 
                        }
                    } else {
                        if (w.type !== "hidden") {
                            w._mrwOldType = w.type;
                            w.type = "hidden";
                        }
                        w.hidden = true;
                    }
                }
                
                setTimeout(() => {
                    const size = node.computeSize();
                    size[1] += 60; // Extra room for the header and padding
                    node.setSize(size);
                    if (node.onResize) node.onResize(node.size);
                    app.graph.setDirtyCanvas(true, true);
                }, 100);
            };

            // Hook into configure and update
            node.onConfigure = function() {
                toggleWidgets();
            };

            // Sync comparison slider
            node.onMouseMove = function(e, pos) {
                if (customUI.showComparison) {
                    const topY = 40; 
                    const previewH = 220;
                    if (pos[1] >= topY && pos[1] <= topY + previewH) {
                        const targetW = this.size[0] - 20;
                        if (pos[0] >= 10 && pos[0] <= 10 + targetW) {
                            customUI.sliderX = (pos[0] - 10) / targetW;
                            this.setDirtyCanvas(true);
                        }
                    }
                }
                if (this._onMouseMove) this._onMouseMove(e, pos);
            };

            // Handle Execution results
            const origOnExecuted = node.onExecuted;
            node.onExecuted = function(output) {
                if (origOnExecuted) origOnExecuted.apply(this, arguments);
                if (output.before?.[0] || output.after?.[0]) {
                    customUI.setImages(output.before?.[0], output.after?.[0]);
                }
            };

            // Set effect callback
            setTimeout(() => {
                const effectWidget = node.widgets.find(w => w.name === "effect");
                if (effectWidget) {
                    const bc = effectWidget.callback;
                    effectWidget.callback = function() {
                        if (bc) bc.apply(this, arguments);
                        toggleWidgets();
                    };
                }
                toggleWidgets();
            }, 500);
        }
    }
});
