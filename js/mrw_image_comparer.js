import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

/**
 * MrWeaz Image Comparer (Pro)
 * A high-performance image comparison node.
 */

function imageDataToUrl(data) {
    if (!data) return "";
    return api.apiURL(`/view?filename=${encodeURIComponent(data.filename)}&type=${data.type}&subfolder=${data.subfolder}${app.getPreviewFormatParam()}${app.getRandParam()}`);
}

class MrWeazImageComparerWidget {
    constructor(name, node) {
        this.name = name;
        this.node = node;
        this.type = "custom";
        this.value = { a: null, b: null };
        this.imgs = { a: new Image(), b: new Image() };
        this.loaded = { a: false, b: false };
        this.sliderX = 0.5; // 0 to 1
        
        // Settings
        this.mode = "Slide"; // Slide, Click
        this.showLabel = true;
    }

    setImages(a_data, b_data) {
        if (a_data) {
            this.value.a = a_data;
            this.loaded.a = false;
            this.imgs.a.onload = () => { this.loaded.a = true; this.node.setDirtyCanvas(true); };
            this.imgs.a.src = imageDataToUrl(a_data);
        }
        if (b_data) {
            this.value.b = b_data;
            this.loaded.b = false;
            this.imgs.b.onload = () => { this.loaded.b = true; this.node.setDirtyCanvas(true); };
            this.imgs.b.src = imageDataToUrl(b_data);
        }
    }

    draw(ctx, node, width, y, widget_offset) {
        const [nodeWidth, nodeHeight] = node.size;
        const topY = y;
        const areaHeight = nodeHeight - topY - 10;
        
        if (areaHeight < 0) return;

        // Background
        ctx.fillStyle = "#000";
        ctx.fillRect(10, topY, width - 20, areaHeight);

        const imgA = this.loaded.a ? this.imgs.a : null;
        const imgB = this.loaded.b ? this.imgs.b : null;

        if (!imgA && !imgB) {
            ctx.fillStyle = "#333";
            ctx.textAlign = "center";
            ctx.fillText("Waiting for images...", width / 2, topY + areaHeight / 2);
            return;
        }

        // Calculate unified target rect (fit both images while maintaining aspect ratio)
        // We assume A and B have the same aspect ratio for best results, 
        // but we'll use A as the master aspect.
        const masterImg = imgA || imgB;
        const aspect = masterImg.naturalWidth / masterImg.naturalHeight;
        
        let targetW = width - 20;
        let targetH = targetW / aspect;
        
        if (targetH > areaHeight) {
            targetH = areaHeight;
            targetW = targetH * aspect;
        }

        const offsetX = (width - targetW) / 2;
        const offsetY = topY + (areaHeight - targetH) / 2;

        ctx.save();
        
        // Draw Image A
        if (imgA) {
            ctx.drawImage(imgA, offsetX, offsetY, targetW, targetH);
        }

        // Draw Image B with clipping if Slide mode
        if (imgB) {
            ctx.beginPath();
            if (this.node.properties.comparer_mode === "Click") {
                if (this.node.mouseOver && app.canvas.pointer_is_down) {
                    ctx.rect(offsetX, offsetY, targetW, targetH);
                } else {
                    ctx.rect(offsetX, offsetY, 0, 0);
                }
            } else {
                // Slide mode
                const splitX = offsetX + (targetW * this.sliderX);
                ctx.rect(splitX, offsetY, targetW - (splitX - offsetX), targetH);
            }
            ctx.clip();
            ctx.drawImage(imgB, offsetX, offsetY, targetW, targetH);
        }

        ctx.restore();

        // Draw Slider Line
        if (this.node.properties.comparer_mode === "Slide" && (imgA || imgB)) {
            const splitX = offsetX + (targetW * this.sliderX);
            ctx.strokeStyle = "#fff";
            ctx.lineWidth = 2;
            ctx.setLineDash([5, 5]);
            ctx.beginPath();
            ctx.moveTo(splitX, offsetY);
            ctx.lineTo(splitX, offsetY + targetH);
            ctx.stroke();
            ctx.setLineDash([]);

            // Handle circle
            ctx.fillStyle = "#fff";
            ctx.beginPath();
            ctx.arc(splitX, offsetY + targetH / 2, 8, 0, Math.PI * 2);
            ctx.fill();
            ctx.fillStyle = "#000";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.font = "bold 8px Arial";
            ctx.fillText("↕", splitX, offsetY + targetH / 2);
        }

        // Labels
        if (this.showLabel) {
            ctx.fillStyle = "rgba(0,0,0,0.5)";
            ctx.fillRect(offsetX + 5, offsetY + 5, 20, 15);
            ctx.fillRect(offsetX + targetW - 25, offsetY + 5, 20, 15);
            ctx.fillStyle = "#fff";
            ctx.font = "bold 10px sans-serif";
            ctx.textAlign = "center";
            ctx.fillText("A", offsetX + 15, offsetY + 16);
            ctx.fillText("B", offsetX + targetW - 15, offsetY + 16);
        }
    }

    computeSize() {
        return [256, 256];
    }
}

app.registerExtension({
    name: "MrWeaz.ImageComparerPro",
    async nodeCreated(node) {
        if (node.comfyClass === "MrWeazImageComparer") {
            node.properties = node.properties || {};
            node.properties.comparer_mode = node.properties.comparer_mode || "Slide";
            
            node.addCustomWidget({
                name: "mode_btn",
                type: "button",
                label: "Switch Mode (Slide/Click)",
                callback: () => {
                    node.properties.comparer_mode = node.properties.comparer_mode === "Slide" ? "Click" : "Slide";
                    node.setDirtyCanvas(true);
                }
            });

            const comparerWidget = new MrWeazImageComparerWidget("comparer", node);
            node.addCustomWidget(comparerWidget);
            node.comparerWidget = comparerWidget;

            node.onMouseMove = function(e, pos) {
                if (this.properties.comparer_mode === "Slide") {
                    const topY = 40; // Approx header + button height
                    const areaHeight = this.size[1] - topY - 10;
                    const targetW = this.size[0] - 20;
                    const offsetX = 10;
                    
                    if (pos[0] >= offsetX && pos[0] <= offsetX + targetW) {
                        comparerWidget.sliderX = (pos[0] - offsetX) / targetW;
                        this.setDirtyCanvas(true);
                    }
                }
            };

            node.getExtraMenuOptions = function(_, options) {
                options.push({
                    content: `Mode: ${this.properties.comparer_mode === "Slide" ? "Click" : "Slide"}`,
                    callback: () => {
                        this.properties.comparer_mode = this.properties.comparer_mode === "Slide" ? "Click" : "Slide";
                        this.setDirtyCanvas(true);
                    }
                }, {
                    content: `${comparerWidget.showLabel ? "Hide" : "Show"} A/B Labels`,
                    callback: () => {
                        comparerWidget.showLabel = !comparerWidget.showLabel;
                        this.setDirtyCanvas(true);
                    }
                });
            };

            const origOnExecuted = node.onExecuted;
            node.onExecuted = function(output) {
                if (origOnExecuted) origOnExecuted.apply(this, arguments);
                
                const a_img = output.a_images?.[0];
                const b_img = output.b_images?.[0];
                
                if (a_img || b_img) {
                    comparerWidget.setImages(a_img, b_img);
                }
            };

            // Size handling
            node.setSize([400, 450]);
        }
    }
});
