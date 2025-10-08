#!/usr/bin/env python3
import os
import re
import io
import json
import uuid
import zipfile
import logging
import time
from logging.handlers import RotatingFileHandler
from datetime import datetime, timezone
from urllib.parse import urlparse, quote
from typing import List, Dict, Optional, Tuple

import requests
from PIL import Image  # NEW
from flask import (
    Flask, render_template, request, jsonify, send_file, abort, g
)

# ========= Config =========
FIGMA_TOKEN = os.environ.get("FIGMA_TOKEN")
if not FIGMA_TOKEN:
    raise RuntimeError("Set FIGMA_TOKEN (export FIGMA_TOKEN=...)")

API = "https://api.figma.com/v1"
HEADERS = {"X-Figma-Token": FIGMA_TOKEN}

RENDERABLE_TYPES = {
    "FRAME", "COMPONENT", "COMPONENT_SET", "INSTANCE", "GROUP",
    "RECTANGLE", "ELLIPSE", "VECTOR", "POLYGON", "STAR", "LINE",
    "TEXT", "SECTION", "SHAPE_WITH_TEXT"
}

# ========= App & Logging =========
app = Flask(__name__)

# ---------- helpers you already had ----------
def slim_tree(node: dict) -> dict:
    out = {
        "id": node.get("id"),
        "name": node.get("name"),
        "type": node.get("type"),
        "children": []
    }
    for c in node.get("children", []) or []:
        out["children"].append(slim_tree(c))
    return out

@app.get("/api/tree")
def api_tree():
    file_key = request.args.get("file_key")
    page_id = request.args.get("page_id")
    if not file_key or not page_id:
        return jsonify({"ok": False, "error": "file_key and page_id are required"}), 400

    nodes_json = fetch_nodes_subtree(file_key, page_id)
    page = nodes_json.get("nodes", {}).get(page_id, {}).get("document")
    if not page or page.get("type") != "CANVAS":
        return jsonify({"ok": False, "error": "Page not found"}), 404

    return jsonify({"ok": True, "tree": slim_tree(page)})

@app.post("/api/thumbs")
def api_thumbs():
    data = request.get_json(force=True, silent=True) or {}
    file_key = data.get("file_key")
    ids = data.get("ids") or []
    fmt = (data.get("format") or "png").lower()
    scale = int(data.get("scale", 2))
    if not file_key or not ids:
        return jsonify({"ok": False, "error": "file_key and ids[] required"}), 400

    urls = images_api_urls(
        file_key, ids, fmt=fmt, scale=scale,
        use_absolute_bounds=True, svg_include_id=True, svg_outline_text=False
    )
    return jsonify({"ok": True, "images": urls})

def setup_logging():
    fmt = "%(asctime)s | %(levelname)s | %(request_id)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    app.logger.setLevel(logging.INFO)
    ch = logging.StreamHandler(); ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
    fh = RotatingFileHandler("app.log", maxBytes=5_000_000, backupCount=3)
    fh.setLevel(logging.INFO); fh.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
    app.logger.handlers.clear(); app.logger.addHandler(ch); app.logger.addHandler(fh)

class RequestIdFilter(logging.Filter):
    def filter(self, record):
        record.request_id = getattr(g, "request_id", "-")
        return True

setup_logging()
for h in app.logger.handlers: h.addFilter(RequestIdFilter())

@app.before_request
def add_request_id():
    g.request_id = request.headers.get("X-Request-Id", str(uuid.uuid4()))

# ========= HTTP helpers & Figma calls =========
def parse_figma_file_key(val: str) -> Optional[str]:
    val = (val or "").trim() if hasattr(str, "trim") else (val or "").strip()
    if not val: return None
    if re.fullmatch(r"[A-Za-z0-9]{10,64}", val): return val
    try:
        u = urlparse(val)
        if not u.netloc.endswith("figma.com"): return None
        parts = [p for p in u.path.split("/") if p]
        for i, p in enumerate(parts):
            if p in ("file", "design", "proto") and i + 1 < len(parts):
                key = parts[i + 1]
                if re.fullmatch(r"[A-Za-z0-9]{10,64}", key): return key
    except Exception:
        return None
    return None

def http_get(url: str, headers: Dict[str, str] = None, timeout: int = 60):
    r = requests.get(url, headers=headers or {}, timeout=timeout)
    if r.status_code != 200:
        app.logger.warning(f"GET {url} -> {r.status_code} {r.text[:200]}")
        abort(r.status_code, r.text)
    return r

def fetch_file_tree(file_key: str, depth: int = 1) -> dict:
    url = f"{API}/files/{file_key}?depth={depth}"
    return http_get(url, HEADERS).json()

def fetch_nodes_subtree(file_key: str, node_id: str) -> dict:
    url = f"{API}/files/{file_key}/nodes?ids={quote(node_id)}"
    return http_get(url, HEADERS, timeout=120).json()

def list_pages(file_json: dict) -> List[dict]:
    return [
        {"id": p["id"], "name": p["name"]}
        for p in file_json.get("document", {}).get("children", [])
        if p.get("type") == "CANVAS"
    ]

def collect_renderable_ids(node: dict, out: List[str]):
    if node.get("type") in RENDERABLE_TYPES and "id" in node:
        out.append(node["id"])
    for c in node.get("children", []) or []:
        collect_renderable_ids(c, out)
    return out

def first_level_frames_and_sections(page_node: dict) -> List[str]:
    ids = []
    for c in page_node.get("children", []) or []:
        if c.get("type") in {"FRAME", "SECTION"}:
            ids.append(c["id"])
    return ids

def images_api_urls(
    file_key: str,
    ids: List[str],
    fmt: str = "png",
    scale: int = 2,
    use_absolute_bounds: bool = True,
    svg_include_id: bool = True,
    svg_simplify_stroke: Optional[bool] = None,
    svg_outline_text: bool = False,
    max_url_len: int = 7000
) -> Dict[str, Optional[str]]:
    if not ids: return {}
    base = f"{API}/images/{file_key}?format={fmt}"
    if fmt in {"png", "jpg"}: base += f"&scale={scale}"
    if use_absolute_bounds: base += "&use_absolute_bounds=true"
    if fmt == "svg":
        if svg_include_id: base += "&svg_include_id=true"
        if svg_outline_text: base += "&svg_outline_text=true"
        if svg_simplify_stroke is not None:
            base += f"&svg_simplify_stroke={'true' if svg_simplify_stroke else 'false'}"

    def chunker(seq, prefix_len):
        batch, length = [], prefix_len
        for _id in seq:
            add = len(_id) + 1
            if batch and length + add > max_url_len:
                yield batch
                batch, length = [_id], prefix_len + len(_id) + 1
            else:
                batch.append(_id); length += add
        if batch: yield batch

    urls = {}
    prefix_len = len(base) + len("&ids=")
    for batch in chunker(ids, prefix_len):
        url = base + "&ids=" + quote(",".join(batch), safe=",:")
        data = http_get(url, HEADERS, timeout=120).json()
        urls.update(data.get("images", {}))
    return urls

# -------- robust nodes fetch you added --------
def fetch_nodes_docs(file_key: str, ids: List[str]) -> Dict[str, dict]:
    docs: Dict[str, dict] = {}
    if not ids: return docs
    CHUNK = 180; MAX_RETRIES = 4; BACKOFF = 0.8
    for i in range(0, len(ids), CHUNK):
        batch = ids[i:i + CHUNK]
        url = f"{API}/files/{file_key}/nodes?ids={quote(','.join(batch), safe=':,')}"
        attempt = 0
        while True:
            r = requests.get(url, headers=HEADERS, timeout=120)
            if r.status_code == 200:
                try: data = r.json()
                except Exception:
                    app.logger.warning("nodes: 200 but invalid JSON; skipping batch"); break
                nodes = data.get("nodes")
                if not isinstance(nodes, dict):
                    app.logger.warning("nodes: 200 but missing/invalid 'nodes'; skipping batch"); break
                for k, v in nodes.items():
                    doc = (v or {}).get("document")
                    if isinstance(doc, dict): docs[k] = doc
                break
            if r.status_code in (429, 500, 502, 503, 504):
                attempt += 1
                if attempt > MAX_RETRIES:
                    app.logger.warning(f"nodes: giving up after {MAX_RETRIES} retries; batch skipped"); break
                delay = BACKOFF * (2 ** (attempt - 1))
                app.logger.info(f"nodes: {r.status_code}; retry {attempt}/{MAX_RETRIES} in {delay:.1f}s")
                time.sleep(delay); continue
            app.logger.warning(f"nodes: GET -> {r.status_code} {r.text[:200]}"); break
    return docs

def collect_descendants(node: dict, out: List[dict]):
    for c in (node.get("children") or []):
        out.append(c); collect_descendants(c, out)

def is_renderable_type(t: str) -> bool:
    return t in RENDERABLE_TYPES

def deepest_leaves(node: dict, out: List[dict]):
    ch = node.get("children") or []
    if not ch: out.append(node); return
    any_renderable_child = False
    for c in ch:
        deepest_leaves(c, out)
        any_renderable_child = True
    if not any_renderable_child and is_renderable_type(node.get("type", "")):
        out.append(node)

def explode_selection(file_key: str, selected_ids: List[str], mode: str) -> List[dict]:
    if not selected_ids: return []
    docs = fetch_nodes_docs(file_key, selected_ids)
    results: List[dict] = []
    for sid in selected_ids:
        root = docs.get(sid)
        if not isinstance(root, dict):
            app.logger.info(f"explode_selection: no doc for {sid}, skip"); continue
        if mode == "descendants":
            bucket = [root]; collect_descendants(root, bucket)
        elif mode == "leaves":
            tmp = [root]; collect_descendants(root, tmp)
            leaves: List[dict] = []; 
            for n in tmp: deepest_leaves(n, leaves)
            bucket = leaves
        else:
            bucket = [root]
        for n in bucket:
            if not is_renderable_type(n.get("type", "")): continue
            results.append({
                "id": n.get("id"),
                "name": n.get("name") or n.get("id"),
                "type": n.get("type") or "",
                "path": [root.get("name") or sid] if n.get("id") != sid else []
            })
    seen, out = set(), []
    for r in results:
        nid = r.get("id")
        if not nid or nid in seen: continue
        seen.add(nid); out.append(r)
    return out

def safe_name(s: str) -> str:
    return re.sub(r'[^A-Za-z0-9._\\-]+', '_', (s or '').strip())[:80]

# ========= Page/Export (existing) =========
@app.get("/")
def index():
    return render_template("index.html", request_id=g.request_id)

@app.get("/api/pages")
def api_pages():
    file_input = request.args.get("file", "")
    file_key = parse_figma_file_key(file_input)
    if not file_key:
        return jsonify({"ok": False, "error": "Invalid Figma link or file key"}), 400
    tree = fetch_file_tree(file_key, depth=1)
    pages = list_pages(tree)
    return jsonify({"ok": True, "file_key": file_key, "pages": pages})

@app.get("/api/page")
def api_page():
    file_key = request.args.get("file_key")
    page_id = request.args.get("page_id")
    fmt = (request.args.get("format") or "png").lower()
    scale = int(request.args.get("scale", 2))
    outlines = request.args.get("outlines", "false").lower() == "true"

    if not file_key or not page_id:
        return jsonify({"ok": False, "error": "file_key and page_id are required"}), 400
    nodes_json = fetch_nodes_subtree(file_key, page_id)
    page = nodes_json.get("nodes", {}).get(page_id, {}).get("document")
    if not page or page.get("type") != "CANVAS":
        return jsonify({"ok": False, "error": "Page not found"}), 404

    preview_ids = first_level_frames_and_sections(page)
    preview_map = images_api_urls(
        file_key, preview_ids, fmt=fmt, scale=scale,
        use_absolute_bounds=True, svg_outline_text=outlines
    )
    all_ids: List[str] = []
    for c in page.get("children", []) or []:
        collect_renderable_ids(c, all_ids)

    return jsonify({
        "ok": True,
        "page": {"id": page_id, "name": page.get("name")},
        "preview": [{"id": nid, "url": preview_map.get(nid)} for nid in preview_ids],
        "all_renderable_ids": all_ids
    })

@app.post("/export")
def export_zip():
    file_key = request.form.get("file_key")
    page_id = request.form.get("page_id")
    fmt = (request.form.get("format") or "png").lower()
    scale = int(request.form.get("scale", 2))
    outlines = request.form.get("outlines", "false").lower() == "true"
    svg_include_id = request.form.get("svg_include_id", "true").lower() == "true"
    svg_simplify = request.form.get("svg_simplify_stroke")
    if svg_simplify is not None and svg_simplify != "":
        svg_simplify = (svg_simplify.lower() == "true")
    else:
        svg_simplify = None
    granularity = (request.form.get("granularity") or "selected").lower()

    ids = request.form.getlist("ids")
    if len(ids) == 1 and "," in ids[0]:
        ids = [s for s in ids[0].split(",") if s.strip()]

    if not file_key or not page_id:
        return jsonify({"ok": False, "error": "file_key and page_id are required"}), 400
    if fmt not in {"png", "jpg", "svg", "pdf"}:
        return jsonify({"ok": False, "error": "format must be png|jpg|svg|pdf"}), 400
    if not ids:
        return jsonify({"ok": False, "error": "No node IDs selected"}), 400

    expanded = explode_selection(file_key, ids, granularity)
    if not expanded:
        return jsonify({"ok": False, "error": "Nothing renderable for chosen granularity"}), 400

    expanded_ids = [x["id"] for x in expanded]
    url_map = images_api_urls(
        file_key, expanded_ids, fmt=fmt, scale=scale,
        use_absolute_bounds=True, svg_include_id=svg_include_id,
        svg_simplify_stroke=svg_simplify, svg_outline_text=outlines
    )
    valid = {k: v for k, v in url_map.items() if v}
    if not valid:
        return jsonify({"ok": False, "error": "Figma returned no exportable URLs (permissions/restrictions?)"}), 403

    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        manifest = []
        for item in expanded:
            nid = item["id"]; url = valid.get(nid)
            if not url: continue
            content = http_get(url, timeout=120).content
            base = safe_name(item["name"]) or nid
            folder = "layers"
            if item["path"]: folder = f"layers/{safe_name(item['path'][0])}"
            ext = fmt; filename = f"{folder}/{base}__{nid}.{ext}"
            zf.writestr(filename, content)
            manifest.append({"id": nid, "name": item["name"], "type": item["type"], "file": filename})
        zf.writestr("manifest.json", json.dumps({
            "file_key": file_key, "page_id": page_id, "format": fmt,
            "granularity": granularity, "count": len(manifest), "items": manifest
        }, indent=2))
    mem.seek(0)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    filename = f"figma-export-{page_id}-{fmt}-{granularity}-{ts}.zip"
    return send_file(mem, as_attachment=True, download_name=filename, mimetype="application/zip")

# ========= NEW: export to disk with optional crop-to-parent =========

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def build_page_index(file_key: str, page_id: str) -> Tuple[dict, Dict[str, dict], Dict[str, str]]:
    """
    Returns (page_doc, id->node map, child->parent map) for the CANVAS subtree.
    """
    js = fetch_nodes_subtree(file_key, page_id)
    page = js.get("nodes", {}).get(page_id, {}).get("document")
    if not page or page.get("type") != "CANVAS":
        abort(404, "Page not found")
    id_map: Dict[str, dict] = {}
    parent: Dict[str, str] = {}
    def walk(n: dict, p: Optional[str]):
        nid = n.get("id")
        if nid: id_map[nid] = n
        if nid and p: parent[nid] = p
        for c in n.get("children", []) or []:
            walk(c, nid)
    walk(page, None)
    return page, id_map, parent

def is_frame_like(t: str) -> bool:
    return t in {"FRAME", "SECTION"}

def find_nearest_parent(id_map: Dict[str, dict], parent: Dict[str, str], node_id: str, mode: str, specific: Optional[str]) -> Optional[str]:
    if mode == "specific" and specific: return specific if specific in id_map else None
    cur = node_id
    while True:
        p = parent.get(cur)
        if not p: return None
        t = (id_map.get(p) or {}).get("type")
        if mode == "nearest_section":
            if t in {"SECTION", "FRAME"}: return p
        else:  # nearest_frame
            if t == "FRAME": return p
        cur = p

def bbox(n: dict, key: str = "absoluteBoundingBox") -> Optional[dict]:
    b = (n or {}).get(key)
    if isinstance(b, dict) and all(k in b for k in ("x","y","width","height")):
        return b
    return None

def compose_child_into_parent_png(child_bytes: bytes, frame_bb: dict, child_bb: dict, scale: int) -> bytes:
    """
    Returns PNG bytes of a frame-sized canvas with the child pasted/clipped at its offset.
    """
    W = int(round(frame_bb["width"]  * scale))
    H = int(round(frame_bb["height"] * scale))
    dx = int(round((child_bb["x"] - frame_bb["x"]) * scale))
    dy = int(round((child_bb["y"] - frame_bb["y"]) * scale))
    im_child = Image.open(io.BytesIO(child_bytes)).convert("RGBA")
    canvas = Image.new("RGBA", (W, H), (0,0,0,0))
    # intersection
    cx0, cy0 = max(0, dx), max(0, dy)
    cx1, cy1 = min(W, dx + im_child.width), min(H, dy + im_child.height)
    if cx1 > cx0 and cy1 > cy0:
        sx0 = cx0 - dx; sy0 = cy0 - dy
        sx1 = sx0 + (cx1 - cx0); sy1 = sy0 + (cy1 - cy0)
        canvas.paste(im_child.crop((sx0, sy0, sx1, sy1)), (cx0, cy0))
    out = io.BytesIO(); canvas.save(out, "PNG"); out.seek(0)
    return out.read()

@app.post("/export_to_disk")
def export_to_disk():
    """
    JSON body:
      file_key, page_id, ids[], format, scale, outlines, svg_include_id, svg_simplify_stroke,
      granularity, save_root, folder_pattern, filename_pattern,
      crop(bool), parent_mode(nearest_frame|nearest_section|specific), parent_id
    """
    data = request.get_json(force=True, silent=True) or {}
    file_key = data.get("file_key"); page_id = data.get("page_id")
    ids = data.get("ids") or []
    fmt = (data.get("format") or "png").lower()
    scale = int(data.get("scale") or 2)
    outlines = bool(data.get("outlines") or False)
    svg_include_id = bool(data.get("svg_include_id") if data.get("svg_include_id") is not None else True)
    svg_simplify = data.get("svg_simplify_stroke")  # True/False/None
    granularity = (data.get("granularity") or "selected").lower()
    save_root = data.get("save_root") or "./output"
    folder_pattern = data.get("folder_pattern") or "{page}/{parent_name}"
    filename_pattern = data.get("filename_pattern") or "{node_name}__{node_id}.{ext}"
    crop = bool(data.get("crop") or False)
    parent_mode = (data.get("parent_mode") or "nearest_frame").lower()
    specific_parent_id = data.get("parent_id")

    if not file_key or not page_id:
        return jsonify({"ok": False, "error": "file_key and page_id are required"}), 400
    if not ids:
        return jsonify({"ok": False, "error": "No node IDs selected"}), 400
    if fmt not in {"png","jpg","svg","pdf"}:
        return jsonify({"ok": False, "error": "format must be png|jpg|svg|pdf"}), 400
    if crop and fmt in {"svg","pdf"}:
        # we only implement raster crop here
        fmt = "png"

    # 1) expand selection per granularity
    expanded = explode_selection(file_key, ids, granularity)
    if not expanded:
        return jsonify({"ok": False, "error": "Nothing renderable for chosen granularity"}), 400
    expanded_ids = [x["id"] for x in expanded]

    # 2) build page index so we can find parents & bounding boxes
    page_doc, id_map, parent_map = build_page_index(file_key, page_id)
    page_name = page_doc.get("name") or page_id

    # 3) export raw node bitmaps (or svg) from Figma
    url_map = images_api_urls(
        file_key, expanded_ids,
        fmt=("png" if crop and fmt in {"png","jpg"} else fmt),
        scale=scale, use_absolute_bounds=True,
        svg_include_id=svg_include_id,
        svg_simplify_stroke=svg_simplify if isinstance(svg_simplify, bool) else None,
        svg_outline_text=outlines
    )

    ensure_dir(save_root)
    saved, errors = [], []

    for item in expanded:
        nid = item["id"]; nurl = url_map.get(nid)
        if not nurl:
            errors.append({"id": nid, "error": "no export URL"}); continue

        node = id_map.get(nid)
        if not node:
            errors.append({"id": nid, "error": "node not in page index"}); continue

        node_name = item["name"] or nid
        parent_name = "root"; parent_id = None
        frame_bb = None

        if crop:
            # choose parent
            pid = find_nearest_parent(id_map, parent_map, nid, parent_mode, specific_parent_id)
            if not pid:
                errors.append({"id": nid, "error": "no suitable parent found"}); continue
            parent_node = id_map.get(pid)
            parent_id = pid
            parent_name = (parent_node or {}).get("name") or pid
            frame_bb = bbox(parent_node, "absoluteBoundingBox")
            child_bb = bbox(node, "absoluteBoundingBox")
            if not frame_bb or not child_bb:
                errors.append({"id": nid, "error": "missing absoluteBoundingBox"}); continue

        # folder and filename
        folder = folder_pattern.format(
            page=safe_name(page_name),
            parent_name=safe_name(parent_name),
            parent_id=safe_name(parent_id or "root")
        )
        folder_path = os.path.join(save_root, folder)
        ensure_dir(folder_path)

        ext = fmt if not crop else "png"  # crop always raster result
        filename = filename_pattern.format(
            node_name=safe_name(node_name), node_id=nid, ext=ext
        )
        out_path = os.path.join(folder_path, filename)

        try:
            content = http_get(nurl, timeout=180).content
            if crop:
                # compose to parent-sized canvas
                composed = compose_child_into_parent_png(content, frame_bb, bbox(node), scale)
                with open(out_path, "wb") as f: f.write(composed)
            else:
                # write raw export
                with open(out_path, "wb") as f: f.write(content)
            saved.append(out_path)
        except Exception as ex:
            errors.append({"id": nid, "error": str(ex)})

    # optional manifest alongside save_root
    manifest = {
        "file_key": file_key,
        "page_id": page_id,
        "page_name": page_name,
        "count": len(saved),
        "saved": saved,
        "errors": errors
    }
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    try:
        with open(os.path.join(save_root, f"manifest-{page_id}-{ts}.json"), "w", encoding="utf-8") as mf:
            json.dump(manifest, mf, indent=2)
    except Exception:
        pass

    return jsonify({"ok": True, "saved_count": len(saved), "errors": errors})

# ========= Error handling =========
@app.errorhandler(400)
@app.errorhandler(403)
@app.errorhandler(404)
@app.errorhandler(500)
def handle_errors(err):
    msg = err.description if hasattr(err, "description") else str(err)
    payload = {"status": getattr(err, "code", 500), "message": msg, "request_id": g.request_id}
    app.logger.error(json.dumps(payload))
    if "text/html" in request.headers.get("Accept", ""):
        return render_template("index.html", error=payload["message"], request_id=g.request_id), payload["status"]
    return jsonify({"ok": False, **payload}), payload["status"]

if __name__ == "__main__":
    # use gunicorn/uvicorn in production
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
