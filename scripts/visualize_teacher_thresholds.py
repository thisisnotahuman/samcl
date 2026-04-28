#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from samcl.data.coco_pairs import CocoPairsDataset
from samcl.teachers.cache import TeacherEmbeddingCache
from samcl.teachers.image_teacher import FrozenImageTeacher, ImageTeacherConfig
from samcl.teachers.text_teacher import FrozenTextTeacher, TextTeacherConfig
from samcl.utils.device import get_device
from samcl.utils.seed import seed_all


def _thumb(in_path: str, out_path: Path, *, size: int = 256) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        return
    im = Image.open(in_path).convert("RGB")
    im.thumbnail((size, size))
    im.save(out_path, format="JPEG", quality=90)


def _dot_all(
    emb: torch.Tensor,  # [N,D], float16/float32 on CPU
    q: np.ndarray,  # [D], float32
    *,
    chunk: int = 16384,
) -> np.ndarray:
    n = emb.shape[0]
    sims = np.empty((n,), dtype=np.float32)
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        # convert chunk to float32 numpy for stable CPU dot
        chunk_np = emb[s:e].cpu().numpy().astype(np.float32, copy=False)
        sims[s:e] = chunk_np @ q
    return sims


def _topk_filtered(sims: np.ndarray, mask: np.ndarray, k: int) -> np.ndarray:
    idx = np.nonzero(mask)[0]
    if idx.size == 0:
        return idx
    k = min(int(k), int(idx.size))
    # partial select then sort
    sub = sims[idx]
    part = np.argpartition(-sub, kth=k - 1)[:k]
    top = idx[part]
    top = top[np.argsort(-sims[top])]
    return top


def _random_from_mask(rng: np.random.Generator, mask: np.ndarray, k: int) -> np.ndarray:
    idx = np.nonzero(mask)[0]
    if idx.size == 0:
        return idx
    k = min(int(k), int(idx.size))
    pick = rng.choice(idx, size=k, replace=False)
    return pick


def main() -> None:
    p = argparse.ArgumentParser("visualize_teacher_thresholds")
    p.add_argument("--coco_images_dir", type=str, required=True)
    p.add_argument("--coco_captions_json", type=str, required=True)
    p.add_argument("--max_pairs", type=int, default=None, help="debug: limit dataset pairs")

    p.add_argument("--cache_dir", type=str, default="./cache", help="same cache_dir as training")
    p.add_argument("--out_dir", type=str, default="./viz_thresholds")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--no_build_cache", action="store_true", help="error if teacher cache missing")

    # teacher config (must match training if you want consistent thresholds)
    p.add_argument("--teacher_text_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--teacher_image_arch", type=str, default="clip", choices=["resnet50", "clip"])
    p.add_argument("--teacher_image_model", type=str, default="openai/clip-vit-base-patch32")

    # thresholds you want to visualize
    p.add_argument("--text_sim_threshold", type=float, default=0.30)
    p.add_argument("--image_sim_threshold", type=float, default=0.65)

    # query controls
    p.add_argument("--query_image_id", type=int, default=None, help="if not set, sample random images")
    p.add_argument("--num_queries", type=int, default=3)

    # display controls
    p.add_argument("--image_topk", type=int, default=12)
    p.add_argument("--image_dissim_k", type=int, default=12)
    p.add_argument("--caption_topk", type=int, default=12)
    p.add_argument("--caption_dissim_k", type=int, default=12)
    p.add_argument("--thumb_size", type=int, default=256)
    p.add_argument("--chunk", type=int, default=16384)
    args = p.parse_args()

    seed_all(int(args.seed))
    rng = np.random.default_rng(int(args.seed))
    device = get_device(args.device)

    out_dir = Path(args.out_dir)
    thumbs_dir = out_dir / "thumbs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) dataset + basic lookups
    dataset = CocoPairsDataset(
        args.coco_images_dir,
        args.coco_captions_json,
        max_pairs=args.max_pairs,
    )
    image_id_to_path: dict[int, str] = {}
    caption_id_to_text: dict[int, str] = {}
    for p_ in dataset.pairs:
        image_id_to_path.setdefault(int(p_.image_id), p_.image_path)
        caption_id_to_text[int(p_.caption_id)] = p_.caption

    # 2) teacher cache (embeddings)
    text_teacher = FrozenTextTeacher(TextTeacherConfig(model_name=args.teacher_text_model), device=device)
    image_teacher = FrozenImageTeacher(
        ImageTeacherConfig(arch=args.teacher_image_arch, clip_model_name=args.teacher_image_model),
        device=device,
    )
    teacher_cache = TeacherEmbeddingCache(
        args.cache_dir,
        dataset=dataset,
        text_teacher=text_teacher,
        image_teacher=image_teacher,
    )
    if args.no_build_cache and (not teacher_cache.load_if_exists()):
        raise RuntimeError(f"Teacher cache not found in {args.cache_dir}. Run training or build cache first.")
    teacher_cache.ensure_built()

    assert teacher_cache.image_ids is not None and teacher_cache.image_emb is not None
    assert teacher_cache.caption_ids is not None and teacher_cache.caption_emb is not None

    img_ids = teacher_cache.image_ids.cpu().numpy().astype(np.int64)
    cap_ids = teacher_cache.caption_ids.cpu().numpy().astype(np.int64)
    img_emb = teacher_cache.image_emb.cpu()
    cap_emb = teacher_cache.caption_emb.cpu()

    img_id_to_row = teacher_cache.image_id_to_row or {}
    cap_id_to_row = teacher_cache.caption_id_to_row or {}

    # choose queries
    if args.query_image_id is not None:
        query_image_ids = [int(args.query_image_id)]
    else:
        query_image_ids = rng.choice(img_ids, size=min(int(args.num_queries), img_ids.shape[0]), replace=False).tolist()

    report = []
    report.append("<html><head><meta charset='utf-8'/>")
    report.append(
        "<style>"
        "body{font-family:Arial, sans-serif; max-width:1200px; margin:24px auto; padding:0 12px;}"
        "h2{margin-top:36px;}"
        ".grid{display:grid; grid-template-columns:repeat(4, 1fr); gap:12px;}"
        ".card{border:1px solid #ddd; border-radius:8px; padding:8px;}"
        ".card img{width:100%; height:auto; border-radius:6px;}"
        ".small{color:#555; font-size:12px;}"
        "pre{background:#f7f7f7; padding:10px; border-radius:8px; overflow:auto;}"
        "</style>"
    )
    report.append("</head><body>")
    report.append("<h1>Teacher threshold visualization</h1>")
    report.append("<pre>")
    report.append(
        html.escape(
            json.dumps(
                {
                    "text_sim_threshold": float(args.text_sim_threshold),
                    "image_sim_threshold": float(args.image_sim_threshold),
                    "teacher_text_model": args.teacher_text_model,
                    "teacher_image_arch": args.teacher_image_arch,
                    "teacher_image_model": args.teacher_image_model,
                    "cache_dir": args.cache_dir,
                },
                indent=2,
            )
        )
    )
    report.append("</pre>")

    for qi, image_id in enumerate(query_image_ids):
        if int(image_id) not in img_id_to_row:
            report.append(f"<h2>Query {qi}: image_id={image_id} (missing in cache)</h2>")
            continue

        qpath = image_id_to_path[int(image_id)]
        qthumb = thumbs_dir / f"query_{qi}_img_{int(image_id)}.jpg"
        _thumb(qpath, qthumb, size=int(args.thumb_size))

        # captions for query image
        q_cap_ids = dataset.caption_ids_for_image(int(image_id))
        q_caps = [caption_id_to_text.get(int(cid), "") for cid in q_cap_ids]
        q_caps_html = "<br/>".join([html.escape(c) for c in q_caps]) if q_caps else "<i>(no captions found)</i>"

        report.append(f"<h2>Query {qi}: image_id={int(image_id)}</h2>")
        report.append("<div class='card'>")
        report.append(f"<img src='{qthumb.relative_to(out_dir)}'/>")
        report.append(f"<div class='small'><b>Captions (GT)</b><br/>{q_caps_html}</div>")
        report.append("</div>")

        # --- image similarity neighbors ---
        q_row = img_id_to_row[int(image_id)]
        q_vec = img_emb[q_row].numpy().astype(np.float32)
        sims_img = _dot_all(img_emb, q_vec, chunk=int(args.chunk))

        not_self = img_ids != int(image_id)
        sim_mask = (sims_img >= float(args.image_sim_threshold)) & not_self
        dis_mask = (sims_img < float(args.image_sim_threshold)) & not_self

        sim_rows = _topk_filtered(sims_img, sim_mask, int(args.image_topk))
        dis_rows = _random_from_mask(rng, dis_mask, int(args.image_dissim_k))

        report.append("<h3>Similar images (teacher cosine ≥ threshold)</h3>")
        report.append("<div class='grid'>")
        for r in sim_rows.tolist():
            iid = int(img_ids[r])
            ipath = image_id_to_path.get(iid)
            if not ipath:
                continue
            ithumb = thumbs_dir / f"q{qi}_sim_img_{iid}.jpg"
            _thumb(ipath, ithumb, size=int(args.thumb_size))
            # show one caption for that image (first)
            cids = dataset.caption_ids_for_image(iid)
            cap = caption_id_to_text.get(int(cids[0]), "") if cids else ""
            report.append("<div class='card'>")
            report.append(f"<img src='{ithumb.relative_to(out_dir)}'/>")
            report.append(
                f"<div class='small'>image_id={iid}<br/>cos={sims_img[r]:.3f}<br/>{html.escape(cap)}</div>"
            )
            report.append("</div>")
        report.append("</div>")

        report.append("<h3>Dissimilar images (teacher cosine &lt; threshold)</h3>")
        report.append("<div class='grid'>")
        for r in dis_rows.tolist():
            iid = int(img_ids[r])
            ipath = image_id_to_path.get(iid)
            if not ipath:
                continue
            ithumb = thumbs_dir / f"q{qi}_dis_img_{iid}.jpg"
            _thumb(ipath, ithumb, size=int(args.thumb_size))
            cids = dataset.caption_ids_for_image(iid)
            cap = caption_id_to_text.get(int(cids[0]), "") if cids else ""
            report.append("<div class='card'>")
            report.append(f"<img src='{ithumb.relative_to(out_dir)}'/>")
            report.append(
                f"<div class='small'>image_id={iid}<br/>cos={sims_img[r]:.3f}<br/>{html.escape(cap)}</div>"
            )
            report.append("</div>")
        report.append("</div>")

        # --- caption similarity neighbors ---
        # pick query caption as the first GT caption if available
        if not q_cap_ids:
            continue
        qcid = int(q_cap_ids[0])
        if qcid not in cap_id_to_row:
            continue
        qcrow = cap_id_to_row[qcid]
        qcap_vec = cap_emb[qcrow].numpy().astype(np.float32)
        sims_cap = _dot_all(cap_emb, qcap_vec, chunk=int(args.chunk))

        not_self_c = cap_ids != qcid
        sim_mask_c = (sims_cap >= float(args.text_sim_threshold)) & not_self_c
        dis_mask_c = (sims_cap < float(args.text_sim_threshold)) & not_self_c

        sim_caps = _topk_filtered(sims_cap, sim_mask_c, int(args.caption_topk))
        dis_caps = _random_from_mask(rng, dis_mask_c, int(args.caption_dissim_k))

        report.append("<h3>Similar captions to query caption (cosine ≥ threshold)</h3>")
        report.append(f"<div class='small'><b>Query caption</b>: {html.escape(caption_id_to_text.get(qcid,''))}</div>")
        report.append("<div class='grid'>")
        for r in sim_caps.tolist():
            cid = int(cap_ids[r])
            txt = caption_id_to_text.get(cid, "")
            iid = int(dataset.image_id_for_caption(cid))
            ipath = image_id_to_path.get(iid)
            ithumb = None
            if ipath:
                ithumb = thumbs_dir / f"q{qi}_sim_cap_{cid}_img_{iid}.jpg"
                _thumb(ipath, ithumb, size=int(args.thumb_size))
            report.append("<div class='card'>")
            if ithumb is not None:
                report.append(f"<img src='{ithumb.relative_to(out_dir)}'/>")
            report.append(f"<div class='small'>caption_id={cid}<br/>cos={sims_cap[r]:.3f}<br/>{html.escape(txt)}</div>")
            report.append("</div>")
        report.append("</div>")

        report.append("<h3>Dissimilar captions to query caption (cosine &lt; threshold)</h3>")
        report.append("<div class='grid'>")
        for r in dis_caps.tolist():
            cid = int(cap_ids[r])
            txt = caption_id_to_text.get(cid, "")
            iid = int(dataset.image_id_for_caption(cid))
            ipath = image_id_to_path.get(iid)
            ithumb = None
            if ipath:
                ithumb = thumbs_dir / f"q{qi}_dis_cap_{cid}_img_{iid}.jpg"
                _thumb(ipath, ithumb, size=int(args.thumb_size))
            report.append("<div class='card'>")
            if ithumb is not None:
                report.append(f"<img src='{ithumb.relative_to(out_dir)}'/>")
            report.append(f"<div class='small'>caption_id={cid}<br/>cos={sims_cap[r]:.3f}<br/>{html.escape(txt)}</div>")
            report.append("</div>")
        report.append("</div>")

    report.append("</body></html>")

    html_path = out_dir / "report.html"
    html_path.write_text("\n".join(report), encoding="utf-8")
    print(f"[ok] wrote {html_path}")
    print(f"[ok] open in browser: file://{html_path.resolve()}")


if __name__ == "__main__":
    main()

