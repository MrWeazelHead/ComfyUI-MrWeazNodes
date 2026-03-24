import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
    name: "MrWeaz.PromptStudioUI",
    async nodeCreated(node) {
        if (node.comfyClass === "MrWeazPromptStudio") {
            
            // Generate unique ID for this node's UI
            const uiId = "mrweaz-ui-" + node.id;
            
            const widgetContainer = document.createElement("div");
            widgetContainer.className = "mrweaz-studio-container";
            widgetContainer.id = uiId;
            widgetContainer.dataset.captureWheel = "true";
            
            // Prevent scrolling canvas when scrolling inside the UI
            widgetContainer.addEventListener("wheel", (e) => e.stopPropagation());
            widgetContainer.addEventListener("mousedown", (e) => e.stopPropagation());
            widgetContainer.addEventListener("pointerdown", (e) => e.stopPropagation());
            widgetContainer.addEventListener("contextmenu", (e) => e.stopPropagation());

            // Add the DOM widget to the node
            const domWidget = node.addDOMWidget("mrweaz_ui", "div", widgetContainer, {
                serialize: false,
                hideOnZoom: false
            });

            widgetContainer.innerHTML = `
                <style>
                    #${uiId} { font-family: sans-serif; color: #ccc; background: #1a1a1a; display: flex; flex-direction: column; height: 100%; min-height: 40px; border-radius: 4px; border: 1px solid #333; overflow: hidden; }
                    #${uiId} .header { display: flex; background: #222; padding: 5px; gap: 5px; align-items: center; border-bottom: 1px solid #444; }
                    #${uiId} .header button { background: #333; color: #eee; border: 1px solid #555; border-radius: 3px; cursor: pointer; padding: 4px 8px; font-size: 11px; }
                    #${uiId} .header button:hover, #${uiId} .header button.active { background: #555; }
                    #${uiId} .content { flex: 1; overflow: hidden; display: flex; flex-direction: column; background: #111; position: relative; }
                    #${uiId}.collapsed .content { display: none; }
                    
                    /* Search Bars */
                    #${uiId} .search-bar-row { display: flex; width: 100%; border-bottom: 1px solid #444; }
                    #${uiId} .search-bar { flex: 1; padding: 6px; background: #222; border: none; color: #fff; box-sizing: border-box; }
                    #${uiId} .folder-filter { max-width: 35%; padding: 6px; background: #1a1a1a; border: none; color: #ccc; border-left: 1px solid #333; display: none; }
                    
                    /* Prompt Area */
                    #${uiId} .pos-separator { padding: 4px 10px; background: #0f1a0f; color: #8f8; font-size: 10px; font-weight: bold; letter-spacing: 1px; border-bottom: 1px solid #050; display: flex; align-items: center; gap: 6px; }
                    #${uiId} #prompt-area { width: 100%; height: calc(60% - 50px); padding: 10px; background: #0f1a0f; color: #cfc; border: none; resize: none; box-sizing: border-box; font-family: monospace; }
                    #${uiId} #prompt-area:focus { outline: none; }
                    #${uiId} .neg-separator { padding: 4px 10px; background: #1a0f0f; color: #f88; font-size: 10px; font-weight: bold; letter-spacing: 1px; border-top: 1px solid #500; border-bottom: 1px solid #500; display: flex; align-items: center; gap: 6px; }
                    #${uiId} #negative-area { width: 100%; height: calc(40% - 30px); min-height: 60px; padding: 10px; background: #1a0f0f; color: #fcc; border: none; resize: none; box-sizing: border-box; font-family: monospace; border-top: 1px solid #400; }
                    #${uiId} #negative-area:focus { outline: none; }

                    /* Grid / Lists */
                    #${uiId} .scroll-view { flex: 1; overflow-y: auto; padding: 8px; gap: 8px; }
                    #${uiId} .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(120px, 1fr)); align-content: start; }
                    #${uiId} .list { display: flex; flex-direction: column; }
                    
                    /* Scrollbar */
                    #${uiId} .scroll-view::-webkit-scrollbar { width: 6px; }
                    #${uiId} .scroll-view::-webkit-scrollbar-track { background: #111; }
                    #${uiId} .scroll-view::-webkit-scrollbar-thumb { background: #444; border-radius: 3px; }
                    
                    /* LoRA Cards */
                    #${uiId} .lora-card { background: #222; border: 1px solid #444; border-radius: 4px; overflow: hidden; cursor: pointer; display: flex; flex-direction: column; transition: border-color 0.1s; min-height: 160px; position: relative; }
                    #${uiId} .lora-card:hover { border-color: #888; }
                    #${uiId} .lora-card.active { border: 2px solid #5a9; background: #1a2a22; }
                    #${uiId} .lora-img { height: 100px; background: #000; display: flex; align-items: center; justify-content: center; overflow: hidden; }
                    #${uiId} .lora-img img { width: 100%; height: 100%; object-fit: cover; }
                    #${uiId} .lora-info { padding: 4px; font-size: 10px; text-align: center; }
                    #${uiId} .lora-name { color: #fff; font-weight: bold; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
                    #${uiId} .lora-trigger-edit { width: 100%; background: transparent; border: none; color: #888; font-size: 10px; text-align: center; outline: none; padding: 2px 0; border-bottom: 1px solid transparent; transition: border 0.2s, color 0.2s; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; display: block; box-sizing: border-box;}
                    #${uiId} .lora-trigger-edit:focus { border-bottom: 1px solid #5a9; color: #fff; text-overflow: clip; overflow: visible;}
                    #${uiId} .lora-actions { display: flex; flex-direction: column; gap: 2px; padding: 4px; background: rgba(0,0,0,0.2); }
                    
                    /* Civitai Sync Button */
                    @keyframes mrw-spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
                    #${uiId} .civitai-sync-btn { position: absolute; top: 4px; left: 4px; width: 22px; height: 22px; background: rgba(0,0,0,0.6); color: white; border-radius: 50%; font-size: 12px; display: flex; align-items: center; justify-content: center; opacity: 0; transition: opacity 0.2s; cursor: pointer; z-index: 10; border: 1px solid #555; }
                    #${uiId} .lora-card:hover .civitai-sync-btn { opacity: 1; }
                    #${uiId} .civitai-sync-btn:hover { background: rgba(0,0,0,0.9); }
                    #${uiId} .civitai-sync-btn.loading { animation: mrw-spin 1s linear infinite; pointer-events: none; opacity: 1; background: #28a; }
                    
                    /* Civitai Outlink */
                    #${uiId} .civitai-outlink-btn { position: absolute; top: 4px; right: 4px; width: 22px; height: 22px; background: rgba(0,0,0,0.6); color: white; border-radius: 50%; font-size: 12px; display: flex; align-items: center; justify-content: center; opacity: 0; transition: background 0.2s, opacity 0.2s; cursor: pointer; z-index: 10; border: 1px solid #555; text-decoration: none; }
                    #${uiId} .lora-card:hover .civitai-outlink-btn { opacity: 1; }
                    #${uiId} .civitai-outlink-btn:hover { background: rgba(0,0,0,0.9); border-color: #5a9; }


                    /* Wildcard Items */
                    #${uiId} .wc-item { padding: 6px; border-bottom: 1px solid #222; cursor: pointer; font-size: 12px; }
                    #${uiId} .wc-item:hover { background: #333; color: white; }

                    /* Preset Items */
                    #${uiId} .preset-item { padding: 8px; border-bottom: 1px solid #222; cursor: pointer; display: flex; flex-direction: column; gap: 4px; }
                    #${uiId} .preset-item:hover { background: #2a2a2a; }
                    #${uiId} .preset-header { display: flex; justify-content: space-between; align-items: center; color: white; font-weight: bold; font-size: 12px; }
                    #${uiId} .preset-del { background: #800; color: white; border: none; padding: 2px 6px; border-radius: 3px; cursor: pointer; font-size: 10px; }
                    #${uiId} .preset-del:hover { background: #f00; }
                    #${uiId} .preset-prompt { color: #888; font-size: 10px; font-style: italic; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
                </style>
                <div class="header">
                    <button id="tab-prompt" class="active">📝 PROMPT</button>
                    <button id="tab-lora">🖼️ LoRAs</button>
                    <button id="tab-wild">🃏 Wildcards</button>
                    <button id="tab-preset">💾 Presets</button>
                    <button id="tab-settings">⚙️ Settings</button>
                    <button id="btn-final-toggle" style="margin-left:auto; background:#223; color:#ada;">👁 Final Prompt</button>
                    <button id="btn-collapse" style="margin-left: 5px;">[ - ]</button>
                </div>
                <!-- Final Prompt Collapsible Area -->
                <div id="final-prompt-area" style="display:none; padding:10px; background:#111; border-bottom:1px solid #333; color:#8d8; font-family:monospace; font-size:11px; max-height:100px; overflow-y:auto; white-space:pre-wrap;">
                    <i>(Waiting for execution to show the resolved prompt...)</i>
                </div>
                <div class="content">
                    <div class="pos-separator" id="pos-separator">⊕ POSITIVE PROMPT</div>
                    <textarea id="prompt-area" spellcheck="false" placeholder="Your prompt here... use __wildcard__"></textarea>
                    <div class="neg-separator" id="neg-separator">⊝ NEGATIVE PROMPT</div>
                    <textarea id="negative-area" spellcheck="false" placeholder="What you DON't want..."></textarea>
                    <div id="view-tools" style="display: none; flex-direction: column; flex: 1; overflow: hidden;">
                        <div class="search-bar-row">
                            <input type="text" id="search-box" class="search-bar" placeholder="Search...">
                            <select id="folder-filter" class="folder-filter">
                                <option value="ALL">All Folders</option>
                            </select>
                        </div>
                        <div id="view-layer" class="scroll-view grid">
                            <!-- Content goes here -->
                        </div>
                    </div>
                </div>
            `;

            // State
            let collapsed = false;
            let currentTab = "prompt"; // prompt, lora, wild, preset
            let loraData = [];
            let wildData = [];
            let presetData = [];
            let activeLoras = {}; // { lora_name: { model_str: 1.0, clip_str: 1.0 } }
            
            let loraFolders = new Set();
            let currentFolder = "ALL";
            let loraDisplayLimit = 30;
            let filteredLorasCache = [];

            // UI Elements
            const container = widgetContainer;
            const btnCollapse = container.querySelector("#btn-collapse");
            const btnFinalToggle = container.querySelector("#btn-final-toggle");
            const finalPromptArea = container.querySelector("#final-prompt-area");
            
            const tabPrompt = container.querySelector("#tab-prompt");
            const tabLora = container.querySelector("#tab-lora");
            const tabWild = container.querySelector("#tab-wild");
            const tabPreset = container.querySelector("#tab-preset");
            const tabSettings = container.querySelector("#tab-settings");
            const promptArea = container.querySelector("#prompt-area");
            const posSeparator = container.querySelector("#pos-separator");
            const negativeArea = container.querySelector("#negative-area");
            const negSeparator = container.querySelector("#neg-separator");
            const viewTools = container.querySelector("#view-tools");
            const searchBox = container.querySelector("#search-box");
            const folderFilter = container.querySelector("#folder-filter");
            const viewLayer = container.querySelector("#view-layer");

            folderFilter.onchange = (e) => {
                currentFolder = e.target.value;
                loraDisplayLimit = 30;
                renderLoras();
                viewLayer.scrollTop = 0;
            };

            // Helper to get/set normal widgets
            const getW = (n) => { const w = node.widgets.find(x => x.name === n); return w ? w.value : null; };
            const setW = (n, v) => { const w = node.widgets.find(x => x.name === n); if (w && v !== undefined && v !== null) w.value = v; app.graph.setDirtyCanvas(true, true); };

            const updateLoraDataWidget = () => {
                const arr = Object.keys(activeLoras).map(k => ({
                    name: k,
                    model_str: activeLoras[k].model_str,
                    clip_str: activeLoras[k].clip_str,
                    use_trigger: activeLoras[k].use_trigger !== false
                }));
                setW("lora_data", JSON.stringify(arr));
            };

            // Render Functions
            const renderLoras = (appendOnly = false) => {
                if (!appendOnly) {
                    viewLayer.className = "scroll-view grid";
                    viewLayer.innerHTML = "";
                    const q = searchBox.value.toLowerCase();
                    
                    let folderFiltered = loraData;
                    if (currentFolder !== "ALL") {
                        if (currentFolder === "ROOT") folderFiltered = loraData.filter(l => l.name.indexOf('/') === -1 && l.name.indexOf('\\') === -1);
                        else folderFiltered = loraData.filter(l => {
                            const normalName = l.name.replaceAll('\\', '/');
                            return normalName.startsWith(currentFolder + '/');
                        });
                    }

                    // Sort active to the top
                    const sortedData = [...folderFiltered].sort((a, b) => {
                        const aAct = activeLoras[a.name] !== undefined;
                        const bAct = activeLoras[b.name] !== undefined;
                        if (aAct && !bAct) return -1;
                        if (!aAct && bAct) return 1;
                        return a.name.localeCompare(b.name);
                    });

                    filteredLorasCache = sortedData.filter(l => l.name.toLowerCase().includes(q) || (l.trigger_words && l.trigger_words.toLowerCase().includes(q)));
                    loraDisplayLimit = 30;
                }
                
                const toRender = appendOnly ? filteredLorasCache.slice(loraDisplayLimit - 30, loraDisplayLimit) : filteredLorasCache.slice(0, loraDisplayLimit);
                
                toRender.forEach(l => {
                    const isActive = activeLoras[l.name] !== undefined;
                    const card = document.createElement("div");
                    card.className = "lora-card" + (isActive ? " active" : "");
                    
                    const img = document.createElement("div");
                    img.className = "lora-img";
                    img.innerHTML = l.preview_url ? `<img src="${l.preview_url}">` : `<span style="color:#555;font-size:10px;">${l.description ? 'Has Info' : 'No Preview'}</span>`;
                    if (l.description) img.title = l.description;
                    
                    const info = document.createElement("div");
                    info.className = "lora-info";
                    info.title = l.name;
                    info.innerHTML = `<div class="lora-name">${l.name.split("/").pop().replace(".safetensors", "")}</div>
                                      <input type="text" spellcheck="false" class="lora-trigger-edit" value="${l.trigger_words || ''}" placeholder="No tags" title="Edit triggers: ${l.trigger_words || 'Empty'}">`;
                    
                    const tEdit = info.querySelector(".lora-trigger-edit");
                    tEdit.onclick = e => e.stopPropagation();
                    tEdit.onkeydown = e => { e.stopPropagation(); if (e.key === 'Enter') tEdit.blur(); };
                    tEdit.onchange = async (e) => {
                        const newTriggers = e.target.value;
                        l.trigger_words = newTriggers;
                        tEdit.title = `Edit triggers: ${newTriggers || 'Empty'}`;
                        try {
                            tEdit.style.color = "#a5f";
                            await fetch("/mrweaz/update_lora_info", {
                                method: "POST", headers: { "Content-Type": "application/json" },
                                body: JSON.stringify({ lora_name: l.name, triggers: newTriggers })
                            });
                            tEdit.style.color = "#5a9";
                            setTimeout(() => tEdit.style.color = "", 1000);
                        } catch (err) { tEdit.style.color = "red"; }
                        
                        // Updates prompt payload immediately
                        if (activeLoras[l.name]) {
                            activeLoras[l.name].custom_triggers = newTriggers;
                            updateLoraDataWidget();
                        }
                    };

                    const syncBtn = document.createElement("div");
                    syncBtn.className = "civitai-sync-btn";
                    syncBtn.innerHTML = "☁️";
                    syncBtn.title = "Sync Metadata from CivitAI";
                    syncBtn.onclick = async (e) => {
                        e.stopPropagation();
                        syncBtn.classList.add("loading");
                        syncBtn.innerHTML = "🔄";
                        try {
                            const res = await fetch("/mrweaz/sync_civitai", {
                                method: "POST", headers: { "Content-Type": "application/json" },
                                body: JSON.stringify({ lora_name: l.name })
                            });
                            const result = await res.json();
                            if (result.status === "success") {
                                l.preview_url = result.preview_url || l.preview_url;
                                l.trigger_words = result.trigger_words || l.trigger_words;
                                l.description = result.description || l.description;
                                l.civitai_url = result.civitai_url || l.civitai_url;

                                img.innerHTML = l.preview_url ? `<img src="${l.preview_url}">` : `<span style="color:#555;font-size:10px;">${l.description ? 'Has Info' : 'No Preview'}</span>`;
                                if (l.description) img.title = l.description;
                                info.innerHTML = `<div class="lora-name">${l.name.split("/").pop().replace(".safetensors", "")}</div>
                                                  <input type="text" spellcheck="false" class="lora-trigger-edit" value="${l.trigger_words || ''}" placeholder="No tags" title="Edit triggers: ${l.trigger_words || 'Empty'}">`;
                                // Rebind listeners on sync-regeneration
                                const tEditNewly = info.querySelector(".lora-trigger-edit");
                                tEditNewly.onclick = e => e.stopPropagation();
                                tEditNewly.onkeydown = e => { e.stopPropagation(); if (e.key === 'Enter') tEditNewly.blur(); };
                                tEditNewly.onchange = async (ev) => {
                                    l.trigger_words = ev.target.value;
                                    tEditNewly.title = `Edit triggers: ${ev.target.value || 'Empty'}`;
                                    try {
                                        tEditNewly.style.color = "#a5f";
                                        await fetch("/mrweaz/update_lora_info", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ lora_name: l.name, triggers: ev.target.value }) });
                                        tEditNewly.style.color = "#5a9"; setTimeout(() => tEditNewly.style.color = "", 1000);
                                    } catch (err) { tEditNewly.style.color = "red"; }
                                    if (activeLoras[l.name]) { activeLoras[l.name].custom_triggers = ev.target.value; updateLoraDataWidget(); }
                                };
                                
                                if (l.civitai_url && !card.querySelector(".civitai-outlink-btn")) {
                                    const outBtn = document.createElement("a");
                                    outBtn.className = "civitai-outlink-btn"; outBtn.innerHTML = "🌐"; outBtn.title = "Open on CivitAI"; outBtn.href = l.civitai_url; outBtn.target = "_blank"; outBtn.onclick = ev => ev.stopPropagation();
                                    card.appendChild(outBtn);
                                }

                                syncBtn.innerHTML = "✔️";
                                setTimeout(() => { syncBtn.innerHTML = "☁️"; }, 2000);
                            } else {
                                syncBtn.innerHTML = "❌";
                                syncBtn.title = result.error || "Failed";
                                setTimeout(() => { syncBtn.innerHTML = "☁️"; syncBtn.title = "Sync Metadata from CivitAI"; }, 3000);
                            }
                        } catch (err) {
                            syncBtn.innerHTML = "❌";
                            setTimeout(() => { syncBtn.innerHTML = "☁️"; }, 3000);
                        } finally {
                            syncBtn.classList.remove("loading");
                        }
                    };

                    const actions = document.createElement("div");
                    actions.className = "lora-actions";
                    
                    const setInactiveHTML = () => {
                        actions.innerHTML = `<div style="text-align:center;color:#666;font-size:9px;padding:4px;">Click to Activate</div>`;
                    };

                    const setActiveHTML = () => {
                        actions.innerHTML = `
                            <div style="display:flex; justify-content:space-between; font-size:9px; align-items:center; color:#ddd;">
                                <span>Model:</span>
                                <input type="number" step="0.05" value="${activeLoras[l.name].model_str}" class="lora-weight l-mod" style="width:40px; background:#111; color:#fff; border:1px solid #444; border-radius:2px;">
                            </div>
                            <div style="display:flex; justify-content:space-between; font-size:9px; align-items:center; margin-top:2px; color:#ddd;">
                                <span>CLIP:</span>
                                <input type="number" step="0.05" value="${activeLoras[l.name].clip_str}" class="lora-weight l-clip" style="width:40px; background:#111; color:#fff; border:1px solid #444; border-radius:2px;">
                            </div>
                            <div style="display:flex; justify-content:space-between; font-size:9px; align-items:center; margin-top:2px; color:#ddd;">
                                <span>Auto Trigger:</span>
                                <input type="checkbox" class="lora-trigger-check" ${activeLoras[l.name].use_trigger !== false ? 'checked' : ''} style="cursor:pointer; margin:0;">
                            </div>
                        `;
                        const mInput = actions.querySelector(".l-mod");
                        const cInput = actions.querySelector(".l-clip");
                        const tCheck = actions.querySelector(".lora-trigger-check");
                        
                        const updateWeights = (e) => {
                            e.stopPropagation();
                            activeLoras[l.name].model_str = parseFloat(mInput.value) || 0;
                            activeLoras[l.name].clip_str = parseFloat(cInput.value) || 0;
                            activeLoras[l.name].use_trigger = tCheck.checked;
                            updateLoraDataWidget();
                        };
                        mInput.oninput = updateWeights; mInput.onclick = e=>e.stopPropagation(); mInput.onkeydown = e=>e.stopPropagation();
                        cInput.oninput = updateWeights; cInput.onclick = e=>e.stopPropagation(); cInput.onkeydown = e=>e.stopPropagation();
                        tCheck.onchange = updateWeights; tCheck.onclick = e=>e.stopPropagation();
                    };

                    if (isActive) setActiveHTML();
                    else setInactiveHTML();

                    card.onclick = () => {
                        if (activeLoras[l.name] !== undefined) {
                            delete activeLoras[l.name];
                            card.classList.remove("active");
                            setInactiveHTML();
                        } else {
                            activeLoras[l.name] = { model_str: 1.0, clip_str: 1.0, use_trigger: true };
                            card.classList.add("active");
                            setActiveHTML();
                        }
                        updateLoraDataWidget();
                        // Note: avoiding renderLoras() to maintain high UI performance!
                    };

                    card.appendChild(syncBtn);
                    
                    if (l.civitai_url) {
                        const outBtn = document.createElement("a");
                        outBtn.className = "civitai-outlink-btn";
                        outBtn.innerHTML = "🌐";
                        outBtn.title = "Open on CivitAI";
                        outBtn.href = l.civitai_url;
                        outBtn.target = "_blank";
                        outBtn.onclick = ev => ev.stopPropagation();
                        card.appendChild(outBtn);
                    }
                    
                    card.appendChild(img);
                    card.appendChild(info);
                    card.appendChild(actions);
                    viewLayer.appendChild(card);
                });
            };

            const renderWildcards = () => {
                viewLayer.className = "scroll-view list";
                viewLayer.innerHTML = "";
                const q = searchBox.value.toLowerCase();
                const filtered = wildData.filter(w => w.toLowerCase().includes(q));
                
                filtered.forEach(w => {
                    const el = document.createElement("div");
                    el.className = "wc-item";
                    el.innerText = `__${w}__`;
                    el.onclick = () => {
                        const p = getW("prompt") || "";
                        const newP = p + (p.endsWith(" ")||p==="" ? "" : ", ") + `__${w}__`;
                        setW("prompt", newP);
                        promptArea.value = newP;
                    };
                    viewLayer.appendChild(el);
                });
            };

            let isSyncing = false;
            let syncCancel = false;

            const renderSettings = () => {
                viewLayer.className = "scroll-view list";
                viewLayer.innerHTML = `
                    <div style="padding: 10px; color: #ccc; font-family: monospace;">
                        <h3 style="margin-top:0; color:#fff; font-family: sans-serif;">Global Scanner</h3>
                        <p style="font-size: 11px; margin-bottom: 12px; line-height: 1.4;">
                            Automatically scan your entire library and securely download missing triggers, thumbnails, and descriptions from CivitAI. Background crawling respects CivitAI's safe rate limits and will skip files already downloaded or explicitly missing from their database.
                        </p>
                        <div style="margin-bottom: 20px;">
                            <button id="mrw-sync-all-btn" style="background:#28a; color:#fff; border:none; padding:8px 12px; border-radius:4px; cursor:pointer; min-width: 200px;">
                                ${isSyncing ? 'Stop Syncing' : 'Sync ALL missing from CivitAI'}
                            </button>
                            <div id="mrw-sync-status" style="margin-top:12px; font-size:12px; color:#aaa; line-height: 1.5;"></div>
                        </div>
                    </div>
                `;
                
                const syncBtn = container.querySelector("#mrw-sync-all-btn");
                const syncStatus = container.querySelector("#mrw-sync-status");
                if (isSyncing) syncBtn.style.background = "#a33";
                
                syncBtn.onclick = async () => {
                    if (isSyncing) {
                        syncCancel = true;
                        syncBtn.innerText = "Stopping...";
                        syncBtn.disabled = true;
                        return;
                    }
                    
                    const queue = loraData.filter(l => !l.civitai_url && !l.description);
                    if (queue.length === 0) {
                        syncStatus.innerHTML = "<span style='color:#5a9'>All LoRAs are already synced!</span>";
                        return;
                    }
                    
                    isSyncing = true;
                    syncCancel = false;
                    syncBtn.innerText = "Stop Syncing";
                    syncBtn.style.background = "#a33";
                    
                    let successCount = 0;
                    let failCount = 0;
                    
                    for (let i = 0; i < queue.length; i++) {
                        if (syncCancel) break;
                        const l = queue[i];
                        syncStatus.innerHTML = `Syncing ${i+1} / ${queue.length}...<br><span style="color:#fff">${l.name.split("/").pop()}</span>`;
                        try {
                            const res = await fetch("/mrweaz/sync_civitai", {
                                method: "POST", headers: { "Content-Type": "application/json" },
                                body: JSON.stringify({ lora_name: l.name })
                            });
                            const result = await res.json();
                            if (result.status === "success") {
                                successCount++;
                                l.civitai_url = result.civitai_url || l.civitai_url;
                                l.preview_url = result.preview_url || l.preview_url;
                                l.trigger_words = result.trigger_words || l.trigger_words;
                                l.description = result.description || l.description;
                            } else {
                                failCount++;
                            }
                        } catch (e) { failCount++; }
                        
                        await new Promise(r => setTimeout(r, 600)); // safe Civitai rate limit
                    }
                    
                    isSyncing = false;
                    syncCancel = false;
                    syncBtn.disabled = false;
                    syncBtn.innerText = "Sync ALL missing from CivitAI";
                    syncBtn.style.background = "#28a";
                    syncStatus.innerHTML = `Finished!<br>Successfully synced: <span style="color:#5a9;font-weight:bold;">${successCount}</span> | Failed or Unlisted: <span style="color:#a55;font-weight:bold;">${failCount}</span>`;
                };
            };

            const renderPresets = () => {
                viewLayer.className = "scroll-view list";
                viewLayer.innerHTML = "";
                
                const saveBtn = document.createElement("button");
                saveBtn.innerText = "➕ Save Current Configuration as New Preset";
                saveBtn.style.padding = "8px";
                saveBtn.style.background = "#2a5";
                saveBtn.style.color = "white";
                saveBtn.style.border = "none";
                saveBtn.style.borderRadius = "4px";
                saveBtn.style.cursor = "pointer";
                saveBtn.style.marginBottom = "10px";
                saveBtn.onclick = async () => {
                    const name = prompt("Name this Preset:");
                    if(!name) return;
                    const presetData = {
                        preset_name: name,
                        timestamp: new Date().toISOString(),
                        prompt: getW("prompt"),
                        seed: getW("seed"),
                        lora_data: getW("lora_data")
                    };
                    await fetch("/mrweaz/save_preset", { method: "POST", body: JSON.stringify(presetData) });
                    fetchData("preset");
                };
                viewLayer.appendChild(saveBtn);

                const q = searchBox.value.toLowerCase();
                const filtered = presetData.filter(p => p.preset_name.toLowerCase().includes(q) || p.prompt.toLowerCase().includes(q));
                
                filtered.forEach(p => {
                    const el = document.createElement("div");
                    el.className = "preset-item";
                    el.innerHTML = `
                        <div class="preset-header">
                            <span>${p.preset_name}</span>
                            <button class="preset-del">❌</button>
                        </div>
                        <div class="preset-prompt">${p.prompt}</div>
                    `;
                    el.onclick = (e) => {
                        if(e.target.className === "preset-del") return;
                        setW("prompt", p.prompt);
                        promptArea.value = p.prompt;
                        setW("seed", p.seed);
                        
                        try {
                            // Restore LoRAs
                            activeLoras = {};
                            const newLoras = JSON.parse(p.lora_data || "[]");
                            newLoras.forEach(l => {
                                activeLoras[l.name] = { model_str: l.model_str, clip_str: l.clip_str };
                            });
                            // Legacy preset compatibility
                            if (p.lora_1 && p.lora_1 !== "None") activeLoras[p.lora_1] = { model_str: p.lora_1_str?.[0]||1, clip_str: p.lora_1_str?.[1]||1 };
                            if (p.lora_2 && p.lora_2 !== "None") activeLoras[p.lora_2] = { model_str: p.lora_2_str?.[0]||1, clip_str: p.lora_2_str?.[1]||1 };
                            if (p.lora_3 && p.lora_3 !== "None") activeLoras[p.lora_3] = { model_str: p.lora_3_str?.[0]||1, clip_str: p.lora_3_str?.[1]||1 };
                            delete activeLoras["None"];
                            updateLoraDataWidget();
                        } catch(err) {}
                    };
                    el.querySelector(".preset-del").onclick = async (e) => {
                        e.stopPropagation();
                        if(confirm(`Delete preset "${p.preset_name}"?`)) {
                            await fetch("/mrweaz/delete_preset", { method: "POST", body: JSON.stringify({ preset_name: p.preset_name }) });
                            fetchData("preset");
                        }
                    };
                    viewLayer.appendChild(el);
                });
            };

            const renderCurrentTab = () => {
                tabPrompt.className = currentTab === "prompt" ? "active" : "";
                tabLora.className = currentTab === "lora" ? "active" : "";
                tabWild.className = currentTab === "wild" ? "active" : "";
                tabPreset.className = currentTab === "preset" ? "active" : "";
                tabSettings.className = currentTab === "settings" ? "active" : "";
                
                if (currentTab === "prompt") {
                    promptArea.style.display = "block";
                    posSeparator.style.display = "flex";
                    negativeArea.style.display = "block";
                    negSeparator.style.display = "flex";
                    viewTools.style.display = "none";
                } else if (currentTab === "settings") {
                    promptArea.style.display = "none";
                    posSeparator.style.display = "none";
                    negativeArea.style.display = "none";
                    negSeparator.style.display = "none";
                    viewTools.style.display = "flex";
                    folderFilter.style.display = "none";
                    searchBox.style.display = "none";
                    renderSettings();
                } else {
                    promptArea.style.display = "none";
                    posSeparator.style.display = "none";
                    negativeArea.style.display = "none";
                    negSeparator.style.display = "none";
                    viewTools.style.display = "flex";
                    searchBox.style.display = "block";
                    searchBox.placeholder = `Search ${currentTab}...`;

                    if(currentTab === "lora") {
                        folderFilter.style.display = loraFolders.size > 0 ? "block" : "none";
                        loraDisplayLimit = 30; // reset scroll limit on tab change
                        renderLoras();
                    } else if(currentTab === "wild") {
                        folderFilter.style.display = "none";
                        renderWildcards();
                    } else if(currentTab === "preset") {
                        folderFilter.style.display = "none";
                        renderPresets();
                    }
                }
            };

            // Infinite Scroll
            viewLayer.onscroll = () => {
                if (currentTab === "lora") {
                    if (viewLayer.scrollHeight - viewLayer.scrollTop - viewLayer.clientHeight < 300) {
                        if (loraDisplayLimit < filteredLorasCache.length) {
                            loraDisplayLimit += 30;
                            renderLoras(true); // append only!
                        }
                    }
                }
            };

            // Fetch Data
            const fetchData = async (tab) => {
                try {
                    if ((tab === "lora" || tab === "settings") && loraData.length === 0) {
                        viewLayer.innerHTML = "<div style='padding:20px;text-align:center;'>Loading LoRAs...</div>";
                        const res = await fetch("/mrweaz/loras"); const data = await res.json(); 
                        loraData = data.loras || [];
                        loraData.forEach(l => {
                            const parts = l.name.split(/[\/\\]/);
                            if (parts.length > 1) loraFolders.add(parts.slice(0, -1).join('/'));
                        });
                        if (loraFolders.size > 0) {
                            folderFilter.innerHTML = '<option value="ALL">All Folders</option><option value="ROOT">/ (Root)</option>' + 
                                Array.from(loraFolders).sort().map(f => `<option value="${f}">${f}</option>`).join("");
                        }
                    } else if (tab === "wild" && wildData.length === 0) {
                        viewLayer.innerHTML = "<div style='padding:20px;text-align:center;'>Loading Wildcards...</div>";
                        const res = await fetch("/mrweaz/wildcards"); const data = await res.json(); wildData = data.wildcards || [];
                    } else if (tab === "preset") {
                        viewLayer.innerHTML = "<div style='padding:20px;text-align:center;'>Loading Presets...</div>";
                        const res = await fetch("/mrweaz/get_presets"); const data = await res.json(); presetData = data.presets || [];
                    }
                    renderCurrentTab();
                } catch(e) { console.error("MrWeaz: Failed to fetch UI data", e); }
            };

            // Event Listeners
            btnCollapse.onclick = () => {
                collapsed = !collapsed;
                container.className = collapsed ? "mrweaz-studio-container collapsed" : "mrweaz-studio-container";
                btnCollapse.innerText = collapsed ? "[ + ]" : "[ - ]";
                
                // Adjust node size heavily based on collapse state
                if (node.size) {
                    if(collapsed) {
                        node.size[1] -= 350; 
                    } else {
                        node.size[1] += 350;
                    }
                    node.setSize(node.size);
                }
                app.graph.setDirtyCanvas(true, true);
            };

            let showingFinal = false;
            btnFinalToggle.onclick = () => {
                showingFinal = !showingFinal;
                finalPromptArea.style.display = showingFinal ? "block" : "none";
                btnFinalToggle.style.background = showingFinal ? "#354" : "#223";
                
                if (node.size) {
                    node.size[1] += showingFinal ? 100 : -100;
                    node.setSize(node.size);
                }
                app.graph.setDirtyCanvas(true, true);
            };

            api.addEventListener("mrweaz.prompt_resolved", (e) => {
                if (e.detail.node == node.id) {
                    finalPromptArea.innerText = e.detail.prompt || "(Empty prompt)";
                }
            });

            tabPrompt.onclick = () => { currentTab = "prompt"; renderCurrentTab(); };
            tabLora.onclick = () => { currentTab = "lora"; searchBox.value = ""; fetchData("lora"); };
            tabWild.onclick = () => { currentTab = "wild"; searchBox.value = ""; fetchData("wild"); };
            tabPreset.onclick = () => { currentTab = "preset"; searchBox.value = ""; fetchData("preset"); };
            tabSettings.onclick = () => { currentTab = "settings"; searchBox.value = ""; fetchData("settings"); };
            
            searchBox.oninput = () => renderCurrentTab();

            // Native Widget synchronization and hiding
            setTimeout(() => {
                promptArea.value = getW("prompt") || "";
                promptArea.addEventListener("input", (e) => {
                    setW("prompt", e.target.value);
                });
                
                negativeArea.value = getW("negative_prompt") || "";
                negativeArea.addEventListener("input", (e) => {
                    setW("negative_prompt", e.target.value);
                });
                negativeArea.addEventListener("keydown", e => e.stopPropagation());
                negativeArea.addEventListener("mousedown", e => e.stopPropagation());
                negativeArea.addEventListener("wheel", e => e.stopPropagation());

                try {
                    const savedLoras = JSON.parse(getW("lora_data") || "[]");
                    savedLoras.forEach(l => {
                        activeLoras[l.name] = { model_str: l.model_str, clip_str: l.clip_str };
                    });
                } catch(e) {}

                // Try to hide the original prompt and lora_data widgets from canvas
                for (let w of node.widgets) {
                    if (w.name === "prompt" || w.name === "lora_data" || w.name === "negative_prompt") {
                        w.type = "hidden";
                        w.computeSize = () => [0, -4];
                        if (w.inputEl) {
                            w.inputEl.style.display = "none";
                        }
                    }
                }

                // Removed UI Re-order logic here because unshifting corrupts ComfyUI's indexing logic
                // and leads to values like `auto_trigger_words` randomly getting populated by `lora_data`!
                
                app.graph.setDirtyCanvas(true, true);
            }, 50);

            // Initial node sizing and loading
            // onConfigure fires when a saved workflow is loaded — this fixes the resize bug on restart
            const origConfigure = node.onConfigure;
            node.onConfigure = function(config) {
                if (origConfigure) origConfigure.apply(this, arguments);
                setTimeout(() => {
                    // Restore prompt/negative from the serialized widget values
                    promptArea.value = getW("prompt") || "";
                    negativeArea.value = getW("negative_prompt") || "";
                    // Ensure node has the right size
                    const sz = node.computeSize();
                    sz[0] = Math.max(sz[0], 500);
                    sz[1] = Math.max(sz[1], 600);
                    node.setSize(sz);
                    app.graph.setDirtyCanvas(true, true);
                }, 100);
            };

            setTimeout(() => {
                const sz = node.computeSize();
                sz[0] = Math.max(sz[0], 500);
                sz[1] = Math.max(sz[1], 600);
                node.setSize(sz);
                app.graph.setDirtyCanvas(true, true);
                
                // Initialize default tab
                renderCurrentTab();
                // Pre-fetch LoRAs quietly
                fetchData("lora");
            }, 100);
        }
    }
});

app.registerExtension({
    name: "MrWeaz.HiresFixKSamplerUI",
    async nodeCreated(node) {
        if (node.comfyClass === "MrWeazHiresFixKSampler") {
            const toggleWidgets = () => {
                const hiresEnable = node.widgets?.find(w => w.name === "enable_hires_fix");
                if (!hiresEnable) return;
                
                const show = hiresEnable.value === "True";
                const targetNames = ["upscale_method", "upscale_factor", "hires_steps", "hires_denoise", "auto_detail_face"];
                
                for (const w of node.widgets) {
                    if (targetNames.includes(w.name)) {
                        if (show) {
                            if (w._mrwHidden) {
                                w.type = w._mrwOriginalType;
                                w.computeSize = w._mrwOriginalComputeSize;
                                w._mrwHidden = false;
                            }
                        } else {
                            if (!w._mrwHidden) {
                                w._mrwOriginalType = w.type;
                                w._mrwOriginalComputeSize = w.computeSize;
                                w.type = "hidden";
                                w.computeSize = () => [0, -4];
                                w._mrwHidden = true;
                            }
                        }
                    }
                }
                
                setTimeout(() => {
                    node.setSize(node.computeSize());
                }, 10);
            };
            
            setTimeout(() => {
                const hiresEnable = node.widgets?.find(w => w.name === "enable_hires_fix");
                if (hiresEnable) {
                    const origCallback = hiresEnable.callback;
                    hiresEnable.callback = function() {
                        if(origCallback) origCallback.apply(this, arguments);
                        toggleWidgets();
                    };
                    toggleWidgets(); // Initial check
                }
            }, 50);
        }
    }
});
