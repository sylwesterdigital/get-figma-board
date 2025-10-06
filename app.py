#!/usr/bin/env python3
import os
import re
import io
import json
import uuid
import zipfile
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from urllib.parse import urlparse, quote
from typing import List, Dict, Optional

import requests
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


# ADD near other helpers
def slim_tree(node: dict) -> dict:
    """Return a compact recursive dict of the node subtree."""
    out = {
        "id": node.get("id"),
        "name": node.get("name"),
        "type": node.get("type"),
        "children": []
    }
    for c in node.get("children", []) or []:
        out["children"].append(slim_tree(c))
    return out

# NEW: return full page subtree (compact)
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

    return jsonify({
        "ok": True,
        "tree": slim_tree(page)  # includes all descendants
    })

# NEW: thumbnails for arbitrary node IDs
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
        file_key,
        ids,
        fmt=fmt,
        scale=scale,
        use_absolute_bounds=True,
        svg_include_id=True,
        svg_outline_text=False
    )
    return jsonify({"ok": True, "images": urls})


def setup_logging():
    fmt = "%(asctime)s | %(levelname)s | %(request_id)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    app.logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))

    fh = RotatingFileHandler("app.log", maxBytes=5_000_000, backupCount=3)
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))

    app.logger.handlers.clear()
    app.logger.addHandler(ch)
    app.logger.addHandler(fh)

class RequestIdFilter(logging.Filter):
    def filter(self, record):
        record.request_id = getattr(g, "request_id", "-")
        return True

setup_logging()
for h in app.logger.handlers:
    h.addFilter(RequestIdFilter())

@app.before_request
def add_request_id():
    g.request_id = request.headers.get("X-Request-Id", str(uuid.uuid4()))

# ========= Helpers =========
def parse_figma_file_key(val: str) -> Optional[str]:
    val = (val or "").strip()
    if not val:
        return None
    # plain key?
    if re.fullmatch(r"[A-Za-z0-9]{10,64}", val):
        return val
    # URL?
    try:
        u = urlparse(val)
        if not u.netloc.endswith("figma.com"):
            return None
        parts = [p for p in u.path.split("/") if p]
        for i, p in enumerate(parts):
            if p in ("file", "design", "proto"):
                if i + 1 < len(parts):
                    key = parts[i + 1]
                    if re.fullmatch(r"[A-Za-z0-9]{10,64}", key):
                        return key
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
    svg_simplify_stroke: Optional[bool] = None,  # None = default
    svg_outline_text: bool = False,
    max_url_len: int = 7000
) -> Dict[str, Optional[str]]:
    if not ids:
        return {}

    base = f"{API}/images/{file_key}?format={fmt}"
    if fmt in {"png", "jpg"}:
        base += f"&scale={scale}"
    if use_absolute_bounds:
        base += "&use_absolute_bounds=true"
    if fmt == "svg":
        if svg_include_id:
            base += "&svg_include_id=true"
        if svg_outline_text:
            base += "&svg_outline_text=true"
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



# --- add imports at top if missing
from collections import deque

# --- helper: batch fetch node documents by id
def fetch_nodes_docs(file_key: str, ids: List[str]) -> Dict[str, dict]:
    docs = {}
    # Figma allows up to ~300 ids; chunk safely
    CHUNK = 180
    for i in range(0, len(ids), CHUNK):
        batch = ids[i:i+CHUNK]
        url = f"{API}/files/{file_key}/nodes?ids={quote(','.join(batch), safe=':,')}"
        data = http_get(url, HEADERS, timeout=120).json()
        docs.update({k: v.get("document") for k, v in data.get("nodes", {}).items()})
    return docs

def collect_descendants(node: dict, out: List[dict]):
    for c in (node.get("children") or []):
        out.append(c)
        collect_descendants(c, out)

def is_renderable_type(t: str) -> bool:
    return t in RENDERABLE_TYPES

def deepest_leaves(node: dict, out: List[dict]):
    ch = node.get("children") or []
    if not ch:
        out.append(node)
        return
    any_renderable_child = False
    for c in ch:
        deepest_leaves(c, out)
        any_renderable_child = True
    # if children aren’t renderable but this is, still include node as leaf
    if not any_renderable_child and is_renderable_type(node.get("type", "")):
        out.append(node)

def explode_selection(file_key: str, selected_ids: List[str], mode: str) -> List[dict]:
    """
    mode: 'selected' | 'descendants' | 'leaves'
    returns list[ {id, name, type, path: [..ancestor names..]} ]
    """
    if not selected_ids:
        return []

    docs = fetch_nodes_docs(file_key, selected_ids)
    results = []

    for sid in selected_ids:
        node = docs.get(sid)
        if not node:
            continue

        bucket = []
        if mode == "selected":
            bucket = [node]
        elif mode == "descendants":
            bucket = [node]
            collect_descendants(node, bucket)
        elif mode == "leaves":
            # get entire subtree then keep deepest leaves
            tmp = [node]
            collect_descendants(node, tmp)
            leaves = []
            for n in tmp:
                deepest_leaves(n, leaves)
            bucket = leaves
        else:
            bucket = [node]

        # annotate with simple path (name chain) using parent-less JSON:
        # just include current node name; path reconstruction beyond one level
        # requires walking parents (not provided); encode type + name.
        for n in bucket:
            if not is_renderable_type(n.get("type", "")):
                continue
            results.append({
                "id": n["id"],
                "name": n.get("name") or n["id"],
                "type": n.get("type", ""),
                "path": [ (node.get("name") or sid) ] if n["id"] != sid else []
            })

    # de-dupe by id preserving order
    seen = set()
    uniq = []
    for r in results:
        if r["id"] in seen: continue
        seen.add(r["id"])
        uniq.append(r)
    return uniq

def safe_name(s: str) -> str:
    return re.sub(r'[^A-Za-z0-9._\-]+', '_', s.strip())[:80]






def stream_zip_from_urls(url_map: Dict[str, str], fmt: str) -> io.BytesIO:
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for node_id, url in url_map.items():
            if not url:
                continue
            content = http_get(url, timeout=120).content
            zf.writestr(f"{node_id}.{fmt}", content)
    mem.seek(0)
    return mem

# ========= Routes (Single-Page UI + JSON APIs) =========
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

    # fetch only this page subtree
    nodes_json = fetch_nodes_subtree(file_key, page_id)
    page = nodes_json.get("nodes", {}).get(page_id, {}).get("document")
    if not page or page.get("type") != "CANVAS":
        return jsonify({"ok": False, "error": "Page not found"}), 404

    # preview only first-level frames/sections to keep it fast
    preview_ids = first_level_frames_and_sections(page)
    preview_map = images_api_urls(
        file_key, preview_ids, fmt=fmt, scale=scale,
        use_absolute_bounds=True, svg_text_outlines=outlines
    )
    # full export IDs (all renderables under page)
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
    svg_simplify = request.form.get("svg_simplify_stroke")  # "true"/"false"/None
    if svg_simplify is not None and svg_simplify != "":
        svg_simplify = (svg_simplify.lower() == "true")
    else:
        svg_simplify = None

    # NEW: layer granularity
    granularity = (request.form.get("granularity") or "selected").lower()
    # valid: selected | descendants | leaves

    ids = request.form.getlist("ids")
    if len(ids) == 1 and "," in ids[0]:
        ids = [s for s in ids[0].split(",") if s.strip()]

    if not file_key or not page_id:
        return jsonify({"ok": False, "error": "file_key and page_id are required"}), 400
    if fmt not in {"png", "jpg", "svg", "pdf"}:
        return jsonify({"ok": False, "error": "format must be png|jpg|svg|pdf"}), 400
    if not ids:
        return jsonify({"ok": False, "error": "No node IDs selected"}), 400

    # explode selected nodes per requested granularity
    # For true “layered” output, recommend fmt='svg' and granularity='leaves'
    expanded = explode_selection(file_key, ids, granularity)
    if not expanded:
        return jsonify({"ok": False, "error": "Nothing renderable for chosen granularity"}), 400

    expanded_ids = [x["id"] for x in expanded]

    url_map = images_api_urls(
        file_key,
        expanded_ids,
        fmt=fmt,
        scale=scale,
        use_absolute_bounds=True,
        svg_include_id=svg_include_id,
        svg_simplify_stroke=svg_simplify,
        svg_outline_text=outlines
    )
    valid = {k: v for k, v in url_map.items() if v}
    if not valid:
        return jsonify({"ok": False, "error": "Figma returned no exportable URLs (permissions/restrictions?)"}), 403

    # build ZIP with hierarchy-based filenames + manifest.json
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        manifest = []
        for item in expanded:
            nid = item["id"]
            url = valid.get(nid)
            if not url:
                continue
            content = http_get(url, timeout=120).content

            base = safe_name(item["name"]) or nid
            folder = "layers"
            if item["path"]:
                folder = f"layers/{safe_name(item['path'][0])}"

            ext = fmt
            filename = f"{folder}/{base}__{nid}.{ext}"
            zf.writestr(filename, content)

            manifest.append({
                "id": nid,
                "name": item["name"],
                "type": item["type"],
                "file": filename
            })

        zf.writestr("manifest.json", json.dumps({
            "file_key": file_key,
            "page_id": page_id,
            "format": fmt,
            "granularity": granularity,
            "count": len(manifest),
            "items": manifest
        }, indent=2))

    mem.seek(0)
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    filename = f"figma-export-{page_id}-{fmt}-{granularity}-{ts}.zip"
    return send_file(mem, as_attachment=True, download_name=filename, mimetype="application/zip")



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
    # Prod tip: use gunicorn/uvicorn in production
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
