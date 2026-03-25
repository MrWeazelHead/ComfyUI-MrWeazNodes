# MrWeaz Nodes - Advanced Prompt Studio
# Advanced CLIP encode with wildcards, LoRAs (w/ trigger words), and presets

import os
import re
import json
import random
import datetime

import folder_paths
import comfy.sd
import comfy.utils
from server import PromptServer
from aiohttp import web
import urllib.parse



# ──────────────────────────────────────────────────────────────
# Wildcard System
# ──────────────────────────────────────────────────────────────

_wildcard_index_cache = None
_wildcard_index_time = 0

def _get_wildcard_dirs():
    """Return a list of directories to scan for wildcard .txt files."""
    comfyui_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    own_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wildcards")

    candidates = [
        own_dir,
        os.path.join(comfyui_root, "wildcards"),
        os.path.join(comfyui_root, "custom_nodes", "ComfyUI-Impact-Pack", "wildcards"),
        os.path.join(comfyui_root, "custom_nodes", "comfyui-easy-use", "wildcards"),
        os.path.join(comfyui_root, "custom_nodes", "ComfyUI-MrWeazNodes", "wildcards"),
    ]
    return [d for d in candidates if os.path.isdir(d)]

def _build_wildcard_index(force=False):
    """Build {name: filepath} mapping from all wildcard directories, with caching."""
    global _wildcard_index_cache, _wildcard_index_time
    now = datetime.datetime.now().timestamp()
    
    # Cache index for 60 seconds unless forced
    if not force and _wildcard_index_cache is not None and (now - _wildcard_index_time) < 60:
        return _wildcard_index_cache

    print(f"[MrWeaz Prompt Studio] Indexing wildcards...")
    index = {}
    for wdir in _get_wildcard_dirs():
        if not os.path.isdir(wdir):
            continue
        for root, _, files in os.walk(wdir):
            for fname in files:
                if fname.endswith(".txt"):
                    rel = os.path.relpath(os.path.join(root, fname), wdir)
                    key = rel.replace(os.sep, "/")
                    if key.endswith(".txt"):
                        key = key[:-4]
                    if key not in index:
                        index[key] = os.path.join(root, fname)
    
    _wildcard_index_cache = index
    _wildcard_index_time = now
    return index

_wildcard_cache = {}
_last_index_time = 0

def _get_wildcard_lines(name):
    """Load wildcard lines from file, with caching."""
    if name in _wildcard_cache:
        return _wildcard_cache[name]

    idx = _build_wildcard_index()
    filepath = idx.get(name)
    if filepath is None:
        return None

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        _wildcard_cache[name] = lines
        return lines
    except Exception:
        return None

def resolve_wildcards(text, seed):
    """Replace __name__ patterns with random lines from wildcard files."""
    if not text or "__" not in text:
        return text
        
    rng = random.Random(seed)

    def _replace(match):
        name = match.group(1)
        lines = _get_wildcard_lines(name)
        if lines:
            return rng.choice(lines)
        return match.group(0)

    # Keep resolving nested wildcards (up to 5 levels deep)
    for _ in range(5):
        new_text = re.sub(r"__([a-zA-Z0-9_/\-]+)__", _replace, text)
        if new_text == text:
            break
        text = new_text
    return text


# ──────────────────────────────────────────────────────────────
# Preset & Logging System
# ──────────────────────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

LOG_FILE = os.path.join(DATA_DIR, "Prompt_Studio_Log.json")
PRESET_FILE = os.path.join(DATA_DIR, "Prompt_Presets.json")
LORA_CACHE_FILE = os.path.join(DATA_DIR, "Lora_Metadata_Cache.json")

def _load_json(path):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return []

def _save_json(path, data):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[MrWeaz Prompt Studio] Failed to save {os.path.basename(path)}: {e}")

def _get_saved_labels():
    """Return list of labels for the recall dropdown (Presets first, then recents)."""
    presets = _load_json(PRESET_FILE)
    recents = _load_json(LOG_FILE)
    
    labels = ["None"]
    
    # Add named presets
    for p in presets:
        name = p.get("preset_name", "Unnamed Preset")
        labels.append(f"[PRESET] {name}")
        
    labels.append("--- Recent Generations ---")
        
    # Add recents
    for entry in reversed(recents[-40:]):
        ts = entry.get("timestamp", "?")
        prompt_preview = entry.get("prompt", "")[:35].replace("\n", " ")
        seed = entry.get("seed", "?")
        labels.append(f"[{ts}] seed:{seed} | {prompt_preview}...")
        
    return labels


# ──────────────────────────────────────────────────────────────
# LoRA Metadata & Trigger Word Extraction
# ──────────────────────────────────────────────────────────────

def extract_trigger_words(lora_path):
    """Safely extract trigger words from safetensors metadata."""
    if not lora_path or not lora_path.endswith(".safetensors"):
        return ""
        
    try:
        # Load just the metadata without tensors to be fast
        _, metadata = comfy.utils.load_torch_file(lora_path, safe_load=True, return_metadata=True)
        if not metadata:
            return ""
            
        # Civitai and Kohya SS tags are usually here
        tags = []
        
        # Method 1: ss_tag_frequency (Kohya standard)
        if "ss_tag_frequency" in metadata:
            try:
                freq = json.loads(metadata["ss_tag_frequency"])
                for folder, folder_tags in freq.items():
                    for tag in folder_tags.keys():
                        tags.append(tag)
            except: pass
            
        # Method 2: modelspec.trigger_phrase
        if "modelspec.trigger_phrase" in metadata:
            return metadata["modelspec.trigger_phrase"]
            
        # If we found tags via Kohya, return top 3 most common/first ones
        if tags:
            return ", ".join(list(dict.fromkeys(tags))[:5])
            
    except Exception as e:
        print(f"[MrWeaz Prompt Studio] Error reading LoRA metadata: {e}")
        
    return ""

# ──────────────────────────────────────────────────────────────
# Visual UI Endpoints
# ──────────────────────────────────────────────────────────────

IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.webp', '.gif']

def _get_preview_url(lora_name):
    lora_path = folder_paths.get_full_path("loras", lora_name)
    if not lora_path:
        return ""
    base_name, _ = os.path.splitext(lora_path)
    for ext in IMAGE_EXTENSIONS:
        if os.path.exists(base_name + ext):
            enc_name = urllib.parse.quote_plus(lora_name)
            enc_file = urllib.parse.quote_plus(os.path.basename(base_name + ext))
            return f"/mrweaz/view_image?filename={enc_file}&lora_name={enc_name}"
    return ""

# Cache to eliminate frontend loading lag
_LORAS_PAYLOAD_CACHE = None
_LORAS_LIST_CACHE = None

@PromptServer.instance.routes.get("/mrweaz/loras")
async def api_get_loras(request):
    global _LORAS_PAYLOAD_CACHE, _LORAS_LIST_CACHE
    try:
        lora_files = folder_paths.get_filename_list("loras")
        
        # In-memory fast cache
        if _LORAS_PAYLOAD_CACHE is not None and _LORAS_LIST_CACHE == lora_files:
            return web.json_response({"loras": _LORAS_PAYLOAD_CACHE})
            
        print(f"[MrWeaz Prompt Studio] Scanning LoRA metadata (optimized)...")
        # Load persistent cache
        persistent_cache = {}
        if os.path.exists(LORA_CACHE_FILE):
             try:
                 with open(LORA_CACHE_FILE, "r", encoding="utf-8") as f:
                     persistent_cache = json.load(f)
             except: pass
        
        loras_data = []
        cache_updated = False
        
        for lora in lora_files:
            lora_path = folder_paths.get_full_path("loras", lora)
            if not lora_path: continue
            
            # Check if cache is still valid
            mtime = os.path.getmtime(lora_path)
            cached_entry = persistent_cache.get(lora)
            
            if cached_entry and cached_entry.get("mtime") == mtime:
                loras_data.append(cached_entry)
                continue
            
            # Cache miss or stale: full scan
            cache_updated = True
            trigger_words = extract_trigger_words(lora_path)
            preview_url = _get_preview_url(lora)
            
            civitai_image = ""
            civitai_desc = ""
            civitai_url = ""
            base_path = os.path.splitext(lora_path)[0]
            info_paths = [f"{base_path}.civitai.info", f"{base_path}.json", f"{base_path}.info"]
            for p in info_paths:
                if os.path.exists(p):
                    try:
                        with open(p, "r", encoding="utf-8") as f:
                            info = json.load(f)
                        if "modelId" in info:
                            civitai_url = f"https://civitai.com/models/{info['modelId']}"
                        if "images" in info and isinstance(info["images"], list) and len(info["images"]) > 0:
                            civitai_image = info["images"][0].get("url", "")
                        if "description" in info and info["description"]:
                            import re
                            desc = re.sub(r'<[^>]+>', '', info["description"])
                            civitai_desc = desc[:250] + "..." if len(desc) > 250 else desc
                            
                        if "user_trainedWords" in info:
                            trigger_words = ", ".join(info["user_trainedWords"])
                        elif "trainedWords" in info and isinstance(info["trainedWords"], list) and len(info["trainedWords"]) > 0:
                            trigger_words = ", ".join(info["trainedWords"])
                        break 
                    except:
                        pass
            
            final_preview = preview_url if preview_url else civitai_image
            
            entry = {
                "name": lora,
                "trigger_words": trigger_words,
                "preview_url": final_preview,
                "description": civitai_desc,
                "civitai_url": civitai_url,
                "mtime": mtime # For cache validation
            }
            loras_data.append(entry)
            persistent_cache[lora] = entry
        
        if cache_updated:
            _save_json(LORA_CACHE_FILE, persistent_cache)
            
        _LORAS_PAYLOAD_CACHE = loras_data
        _LORAS_LIST_CACHE = lora_files
        return web.json_response({"loras": loras_data})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return web.json_response({"error": str(e)}, status=500)

import hashlib

def get_file_sha256(filepath):
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

@PromptServer.instance.routes.post("/mrweaz/update_lora_info")
async def api_update_lora_info(request):
    try:
        data = await request.json()
        lora_name = data.get("lora_name")
        triggers = data.get("triggers", "")
        if not lora_name: return web.json_response({"error": "No lora specified"}, status=400)
        
        lora_path = folder_paths.get_full_path("loras", lora_name)
        if not lora_path: return web.json_response({"error": "File not found"}, status=404)
        
        base_path = os.path.splitext(lora_path)[0]
        import os, json
        info_path = f"{base_path}.civitai.info"
        if not os.path.exists(info_path) and os.path.exists(f"{base_path}.json"):
            info_path = f"{base_path}.json"
            
        metadata = {}
        if os.path.exists(info_path):
            try:
                with open(info_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
            except: pass
            
        words = [x.strip() for x in triggers.split(",") if x.strip()]
        metadata["trainedWords"] = words
        metadata["user_trainedWords"] = words
        
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)
            
        global _LORAS_PAYLOAD_CACHE, _LORAS_LIST_CACHE
        _LORAS_PAYLOAD_CACHE = None
        _LORAS_LIST_CACHE = None
        
        return web.json_response({"status": "success"})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)

@PromptServer.instance.routes.post("/mrweaz/sync_civitai")
async def api_sync_civitai(request):
    global _LORAS_PAYLOAD_CACHE, _LORAS_LIST_CACHE
    try:
        data = await request.json()
        lora_name = data.get("lora_name")
        if not lora_name: return web.json_response({"error": "No lora specified"}, status=400)
        
        lora_path = folder_paths.get_full_path("loras", lora_name)
        if not lora_path or not os.path.exists(lora_path): 
            return web.json_response({"error": "File not found"}, status=404)
            
        file_hash = get_file_sha256(lora_path)
        
        req = urllib.request.Request(
            f"https://civitai.com/api/v1/model-versions/by-hash/{file_hash}",
            headers={'User-Agent': 'Mozilla/5.0'}
        )
        try:
            with urllib.request.urlopen(req) as response:
                metadata = json.loads(response.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 404:
                base_path = os.path.splitext(lora_path)[0]
                json_path = f"{base_path}.civitai.info"
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump({"notFound": True, "description": "Not found on CivitAI"}, f, indent=4)
                return web.json_response({"error": "Not found on CivitAI. Ignored."}, status=404)
            return web.json_response({"error": f"Civitai API error: {e.code}"}, status=404)
            
        base_path = os.path.splitext(lora_path)[0]
        json_path = f"{base_path}.civitai.info"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)
            
        # Clear cache so next gallery ping gets updated info natively
        _LORAS_PAYLOAD_CACHE = None
        _LORAS_LIST_CACHE = None
        
        civitai_image = ""
        civitai_desc = ""
        trigger_words = ""
        civitai_url = ""
        
        if "images" in metadata and isinstance(metadata["images"], list) and len(metadata["images"]) > 0:
            civitai_image = metadata["images"][0].get("url", "")
        if "description" in metadata and metadata["description"]:
            import re
            desc = metadata.get("description", "") or ""
            desc = re.sub(r'<[^>]+>', '', desc)
            civitai_desc = desc[:250] + "..." if len(desc) > 250 else desc
        if "trainedWords" in metadata and isinstance(metadata["trainedWords"], list):
            trigger_words = ", ".join(metadata["trainedWords"])
        if "modelId" in metadata:
            civitai_url = f"https://civitai.com/models/{metadata['modelId']}"
            
        return web.json_response({
            "status": "success",
            "preview_url": civitai_image,
            "description": civitai_desc,
            "trigger_words": trigger_words,
            "civitai_url": civitai_url
        })
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)

@PromptServer.instance.routes.get("/mrweaz/wildcards")
async def api_get_wildcards(request):
    try:
        idx = _build_wildcard_index()
        return web.json_response({"wildcards": sorted(list(idx.keys()))})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)

@PromptServer.instance.routes.get("/mrweaz/view_image")
async def api_view_image(request):
    filename = request.query.get('filename')
    lora_name = request.query.get('lora_name')
    if not filename or not lora_name or ".." in filename or "/" in filename or "\\" in filename:
        return web.Response(status=403)
        
    try:
        lora_name_decoded = urllib.parse.unquote_plus(lora_name)
        filename_decoded = urllib.parse.unquote_plus(filename)

        lora_full_path = folder_paths.get_full_path("loras", lora_name_decoded)
        if not lora_full_path:
            return web.Response(status=404, text=f"Lora not found.")
        
        image_path = os.path.join(os.path.dirname(lora_full_path), filename_decoded)
        if os.path.exists(image_path):
            return web.FileResponse(image_path)
        else:
            return web.Response(status=404, text=f"Preview not found.")
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)

@PromptServer.instance.routes.get("/mrweaz/get_presets")
async def api_get_presets(request):
    try:
        presets = _load_json(PRESET_FILE)
        return web.json_response({"presets": presets})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)

@PromptServer.instance.routes.post("/mrweaz/save_preset")
async def api_save_preset(request):
    try:
        data = await request.json()
        preset_name = data.get("preset_name")
        if not preset_name:
            return web.json_response({"error": "Missing preset name"}, status=400)
            
        presets = _load_json(PRESET_FILE)
        # Remove existing preset with same name if it exists to overwrite
        presets = [p for p in presets if p.get("preset_name") != preset_name]
        presets.append(data)
        _save_json(PRESET_FILE, presets)
        
        return web.json_response({"status": "ok"})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)

@PromptServer.instance.routes.post("/mrweaz/delete_preset")
async def api_delete_preset(request):
    try:
        data = await request.json()
        preset_name = data.get("preset_name")
        if not preset_name:
            return web.json_response({"error": "Missing preset name"}, status=400)
            
        presets = _load_json(PRESET_FILE)
        presets = [p for p in presets if p.get("preset_name") != preset_name]
        _save_json(PRESET_FILE, presets)
        
        return web.json_response({"status": "ok"})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


# ──────────────────────────────────────────────────────────────
# Advanced Prompt Studio Node
# ──────────────────────────────────────────────────────────────


class MrWeazPromptStudioAdv:
    """Advanced CLIP encode with wildcards, 3 LoRAs, trigger words, and presets."""

    def __init__(self):
        self.loaded_loras = {}

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP",),
                "model": ("MODEL",),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True, "placeholder": "Your prompt here... use __wildcard__"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "lora_data": ("STRING", {"default": "[]"}),
                "negative_prompt": ("STRING", {"multiline": True, "placeholder": "Negative prompt...", "default": ""}),
                # Power Features
                "auto_trigger_words": (["False", "True"], {"default": "True"}),
                "save_preset_name": ("STRING", {"default": ""}),
                "log_generation": (["False", "True"], {"default": "True"}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"}
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "MODEL", "CLIP", "STRING", "INT", "INT")
    RETURN_NAMES = ("positive", "negative", "model", "clip", "final_prompt", "seed", "token_count")
    FUNCTION = "process"
    CATEGORY = "MrWeazNodes/Klein"

    def _apply_lora(self, model, clip, lora_name, model_str, clip_str):
        if lora_name == "None" or (model_str == 0 and clip_str == 0):
            return model, clip

        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        lora = self.loaded_loras.get(lora_path)
        
        if lora is None:
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            self.loaded_loras[lora_path] = lora

        model, clip = comfy.sd.load_lora_for_models(model, clip, lora, model_str, clip_str)
        return model, clip
        
    def _get_trigger(self, lora_name):
        if lora_name == "None":
            return ""
        path = folder_paths.get_full_path_or_raise("loras", lora_name)
        tw = extract_trigger_words(path)
        
        import os, json
        base_path = os.path.splitext(path)[0]
        info_paths = [f"{base_path}.civitai.info", f"{base_path}.json", f"{base_path}.info"]
        for p in info_paths:
            if os.path.exists(p):
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        info = json.load(f)
                    
                    if "user_trainedWords" in info:
                        return ", ".join(info["user_trainedWords"])
                    elif "trainedWords" in info and isinstance(info["trainedWords"], list) and len(info["trainedWords"]) > 0:
                        return ", ".join(info["trainedWords"])
                except:
                    pass
        return tw

    def process(self, clip, model, prompt, seed,
                lora_data, negative_prompt,
                auto_trigger_words, save_preset_name, log_generation, unique_id=None):

        # parse active loras from JSON
        try:
            active_loras = json.loads(lora_data)
            if not isinstance(active_loras, list):
                active_loras = []
        except:
            active_loras = []

        # 1. Resolve wildcards
        resolved_prompt = resolve_wildcards(prompt, seed)
        
        # 2. Add trigger words if requested
        triggers = []
        if auto_trigger_words == "True":
            for l_info in active_loras:
                if l_info.get("use_trigger", True):
                    if "custom_triggers" in l_info and isinstance(l_info["custom_triggers"], str):
                        t = l_info["custom_triggers"]
                    else:
                        t = self._get_trigger(l_info.get("name", "None"))
                    if t: triggers.append(t)
            
        if triggers:
            resolved_prompt = ", ".join(triggers) + ", " + resolved_prompt

        # 3. Apply LoRAs
        for l_info in active_loras:
            name = l_info.get("name", "None")
            m_str = float(l_info.get("model_str", 1.0))
            c_str = float(l_info.get("clip_str", 1.0))
            model, clip = self._apply_lora(model, clip, name, m_str, c_str)

        # 4. CLIP encode & count tokens (positive)
        tokens = clip.tokenize(resolved_prompt)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        
        # Encode negative prompt
        neg_tokens = clip.tokenize(negative_prompt if negative_prompt else "")
        neg_conditioning = clip.encode_from_tokens_scheduled(neg_tokens)
        
        # Estimate token count (length of tokenized keys across the clip model)
        token_count = 0
        for k in tokens:
            if isinstance(tokens[k], list) and len(tokens[k]) > 0:
                token_count = max(token_count, len(tokens[k][0]))

        # 5. Logging / Presets
        entry = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "seed": seed,
            "prompt": prompt, # Save original with wildcards intact
            "resolved": resolved_prompt,
            "lora_data": lora_data,
        }
        for i, l_info in enumerate(active_loras[:3]):
            entry[f"lora_{i+1}"] = l_info.get("name", "None")
            entry[f"lora_{i+1}_str"] = (l_info.get("model_str", 1.0), l_info.get("clip_str", 1.0))
        
        entry = {k: v for k, v in entry.items() if v is not None}
        
        if save_preset_name.strip():
            presets = _load_json(PRESET_FILE)
            entry["preset_name"] = save_preset_name.strip()
            # Remove old preset with same name
            presets = [p for p in presets if p.get("preset_name") != save_preset_name.strip()]
            presets.append(entry)
            _save_json(PRESET_FILE, presets)
            print(f"[MrWeaz Prompt Studio] Saved preset: {save_preset_name}")
            
        if log_generation == "True":
            recents = _load_json(LOG_FILE)
            recents.append(entry)
            _save_json(LOG_FILE, recents[-100:]) # Keep last 100

        if unique_id is not None:
            PromptServer.instance.send_sync("mrweaz.prompt_resolved", {"node": unique_id, "prompt": resolved_prompt})

        return (conditioning, neg_conditioning, model, clip, resolved_prompt, seed, token_count)

# ──────────────────────────────────────────────────────────────
# Registration
# ──────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "MrWeazPromptStudio": MrWeazPromptStudioAdv,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MrWeazPromptStudio": "(MRW) Advanced Prompt Studio",
}
