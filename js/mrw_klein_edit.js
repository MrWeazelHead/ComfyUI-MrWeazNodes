import { app } from "../../scripts/app.js";

function hideWidget(widget) {
    if (!widget) return;
    widget.hidden = true;
    widget.computeSize = () => [0, -4];
    widget.serialize = false;
    if (typeof widget.type === "string") {
        widget.type = "hidden";
    }
}

app.registerExtension({
    name: "MrWeaz.KleinEditCleanup",
    async nodeCreated(node) {
        if (node.comfyClass !== "MrWeazKleinRefLoader" && node.comfyClass !== "MrWeazKleinReference2") return;

        for (const widget of node.widgets || []) {
            const name = String(widget?.name || "").toLowerCase();
            const label = String(widget?.label || "").toLowerCase();
            if (
                name.includes("auto-refresh after generation") ||
                label.includes("auto-refresh after generation")
            ) {
                hideWidget(widget);
            }
        }
    },
});
