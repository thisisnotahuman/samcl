from __future__ import annotations

import argparse
from dataclasses import asdict
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from samcl.data.coco_pairs import CocoPairsDataset
from samcl.data.wds_pairs import WdsPairsDataset
from samcl.data.collate import ClipCollator, CvclCollator
from samcl.data.cvcl_vocab import build_cvcl_vocab_from_dataset
from samcl.data.saycam_pairs import SayCamPairsDataset
from samcl.eval.four_afc import default_four_afc_metadata_path, evaluate_four_afc
from samcl.eval.retrieval import evaluate_retrieval
from samcl.losses.clip_infonce import clip_infonce_loss
from samcl.losses.siglip_pairwise import siglip_pairwise_loss
from samcl.models.unimodal_dual_encoder import (
    UniModalDualEncoder,
    UniModalDualEncoderConfig,
    load_unimodal_processors,
)
from samcl.sampling.batch_samplers import BinaryMix, RandomBatchSampler, SemanticBatchSampler, SemanticMix
from samcl.semantic.relations import SemanticRelationConfig, SemanticRelationOracle
from samcl.teachers.cache import TeacherEmbeddingCache
from samcl.teachers.image_teacher import FrozenImageTeacher, ImageTeacherConfig
from samcl.teachers.student_mirrored import StudentMirroredImageTeacher, StudentMirroredTextTeacher
from samcl.teachers.text_teacher import FrozenTextTeacher, TextTeacherConfig
from samcl.utils.device import get_device
from samcl.utils.gpu_burn import GpuMatmulBurner
from samcl.utils.logging import append_jsonl
from samcl.utils.seed import seed_all


def student_teacher_cache_tag(args: argparse.Namespace) -> str:
    return "|".join(
        [
            "student_mirrored",
            f"v={args.vision_model_name}",
            f"t={args.text_model_name}",
            f"pd={args.proj_dim}",
            f"ph={args.proj_hidden_dim}",
            f"pl={args.proj_layers}",
            f"vt={int(args.vision_from_scratch)}",
            f"tt={int(args.text_from_scratch)}",
            f"it={float(args.init_temperature)}",
            f"mtl={int(args.max_text_len)}",
        ]
    )


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("samcl.train")

    # Data
    p.add_argument(
        "--dataset_type",
        type=str,
        choices=["coco", "saycam", "cc3m_wds"],
        default="coco",
        help="Dataset: coco (default), saycam, or cc3m_wds (WebDataset .tar shards)",
    )
    p.add_argument("--coco_images_dir", type=str, default=None, help="Required when dataset_type=coco")
    p.add_argument("--coco_captions_json", type=str, default=None, help="Required when dataset_type=coco")
    p.add_argument("--max_pairs", type=int, default=None, help="debug: limit #pairs")
    p.add_argument("--wds_root", type=str, default=None, help="Required when dataset_type=cc3m_wds: shard directory")
    p.add_argument(
        "--wds_shard_glob",
        type=str,
        default="cc3m-train-*.tar",
        help="Glob under wds_root for training shards (default: cc3m-train-*.tar)",
    )
    p.add_argument(
        "--wds_rebuild_index",
        action="store_true",
        help="Rebuild cached WDS pair index under <wds_root>/.samcl_wds_index/",
    )
    # SayCam (when dataset_type=saycam)
    p.add_argument("--saycam_images_root", type=str, default=None, help="SayCam frame images root (e.g. .../5fps)")
    p.add_argument("--saycam_metadata_json", type=str, default=None, help="SayCam expand_correct_seg.json path")
    p.add_argument(
        "--saycam_mode",
        type=str,
        choices=["one_frame", "all_frames"],
        default="one_frame",
        help="one_frame: one random frame per utterance; all_frames: keep all (frame,utterance) pairs",
    )

    # Main model
    p.add_argument(
        "--student_arch",
        type=str,
        default="hf_unimodal",
        choices=["hf_unimodal", "cvcl", "cvcl_vision_hf_text"],
        help=(
            "Student model family. "
            "hf_unimodal=HF AutoModel vision+text (default). "
            "cvcl=CVCL-style (SayCam-pretrained DINO ViT + embedding text). "
            "cvcl_vision_hf_text=CVCL vision (DINO ViT) + HF text (e.g. BERT)."
        ),
    )
    p.add_argument("--vision_model_name", type=str, default="google/vit-base-patch16-224-in21k")
    p.add_argument("--text_model_name", type=str, default="bert-base-uncased")
    p.add_argument(
        "--vision_from_scratch",
        action="store_true",
        help="if set, initialize vision encoder randomly (do not load pretrained weights)",
    )
    p.add_argument(
        "--text_from_scratch",
        action="store_true",
        help="if set, initialize text encoder randomly (do not load pretrained weights)",
    )
    p.add_argument("--proj_dim", type=int, default=256)
    p.add_argument("--proj_hidden_dim", type=int, default=512)
    p.add_argument("--proj_layers", type=int, default=2, choices=[1, 2])
    p.add_argument("--init_temperature", type=float, default=0.07)
    p.add_argument(
        "--pairwise_loss",
        type=str,
        choices=["infonce", "siglip"],
        default="infonce",
        help="Contrastive objective on in-batch pairs: CLIP symmetric InfoNCE (default) or SigLIP sigmoid (needs hf_unimodal).",
    )
    p.add_argument("--max_text_len", type=int, default=77)

    # CVCL student knobs (only used when --student_arch=cvcl)
    p.add_argument("--cvcl_vision_model_name", type=str, default="dino_sfp_vitb16")
    p.add_argument("--cvcl_vision_from_scratch", action="store_true", help="if set, do NOT load CVCL pretrained weights")
    p.add_argument("--cvcl_vocab_max", type=int, default=50000)
    p.add_argument("--cvcl_vocab_min_freq", type=int, default=1)
    p.add_argument("--cvcl_image_size", type=int, default=224)
    p.add_argument("--cvcl_use_strong_aug", action="store_true", help="use CVCL strong image augmentation (RandomResizedCrop etc.)")

    # Training
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--deterministic", action="store_true", help="try to make results deterministic (may be slower)")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--pin_memory", action="store_true")
    p.add_argument(
        "--prefetch_factor",
        type=int,
        default=2,
        help="DataLoader prefetch_factor (per worker; only if num_workers>0). PyTorch requires >=2. Try 4–8 if CPU/RAM allow to overlap decode/I/O with training.",
    )
    p.add_argument(
        "--dataloader_worker_threads",
        type=int,
        default=1,
        help="torch.set_num_threads in each DataLoader worker. Use 1 when num_workers>1 to reduce OpenMP/MKL CPU oversubscription (often speeds JPEG+HF preprocess).",
    )
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_steps", type=int, default=2000)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument(
        "--save_every",
        type=int,
        default=1000,
        help="checkpoint save interval (steps). 0 disables periodic saves (still saves at end).",
    )
    p.add_argument(
        "--save_latest_only",
        action="store_true",
        help="if set, only write checkpoints/latest.pt (no step_*.pt snapshots on periodic saves).",
    )
    p.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="path to checkpoint (.pt) to resume from",
    )
    p.add_argument(
        "--auto_resume",
        action="store_true",
        help="if set, resume from --run_dir/checkpoints/latest.pt when it exists",
    )
    p.add_argument(
        "--resume_strict",
        action="store_true",
        help="attempt to resume at exact dataloader position by skipping batches (can be VERY slow for expensive samplers)",
    )
    p.add_argument(
        "--gpu_burn_every",
        type=int,
        default=0,
        help="HPC workaround: if >0, run extra GPU matmuls every N steps to increase utilization (0 disables).",
    )
    p.add_argument("--gpu_burn_matmul_dim", type=int, default=2048, help="HPC workaround: matmul matrix dim")
    p.add_argument("--gpu_burn_iters", type=int, default=4, help="HPC workaround: number of chained matmuls")
    p.add_argument(
        "--gpu_burn_dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="HPC workaround: dtype for burn matmuls",
    )
    p.add_argument(
        "--gpu_burn_reserve_gb",
        type=float,
        default=0.0,
        help="HPC workaround: reserve this many GB of VRAM via a persistent buffer (0 disables).",
    )

    # Sampling strategy (main research variable)
    p.add_argument("--sampling_strategy", type=str, choices=["random", "semantic"], default="random")
    p.add_argument(
        "--semantic_relation_mode",
        type=str,
        default="full",
        choices=["full", "text_only", "image_only"],
        help="relation definition used by semantic sampling (4-way full, or 2-way text/image only)",
    )
    p.add_argument("--mix_r1", type=float, default=0.25)
    p.add_argument("--mix_r2", type=float, default=0.25)
    p.add_argument("--mix_r3", type=float, default=0.25)
    p.add_argument("--mix_r4", type=float, default=0.25)
    p.add_argument(
        "--mix_similar",
        type=float,
        default=0.5,
        help="(relation_mode=text_only|image_only) prob of sampling the 'similar' relation",
    )
    p.add_argument(
        "--mix_different",
        type=float,
        default=0.5,
        help="(relation_mode=text_only|image_only) prob of sampling the 'different' relation",
    )
    p.add_argument(
        "--sampler_max_tries",
        type=int,
        default=1000,
        help="semantic sampler: max rejection-sampling tries per sample before fallback to random",
    )
    p.add_argument(
        "--semantic_sampler_mode",
        type=str,
        default="single",
        choices=["single", "multi", "global", "block_global", "c", "b", "a", "bg"],
        help="semantic sampler mode: single/c (scheme C), multi/b (scheme B), global/a (scheme A), block_global/bg (K-block global continuum)",
    )
    p.add_argument(
        "--semantic_sampler_num_anchors",
        type=int,
        default=4,
        help="(mode=multi) number of anchors to check per candidate",
    )
    p.add_argument(
        "--semantic_sampler_min_anchor_matches",
        type=int,
        default=None,
        help="(mode=multi) accept candidate if >= this many anchors match the target relation; default=majority",
    )
    p.add_argument(
        "--semantic_sampler_global_num_candidates",
        type=int,
        default=64,
        help="(mode=global) number of random proposals to score per added sample",
    )
    p.add_argument(
        "--semantic_sampler_num_blocks",
        type=int,
        default=1,
        help="(mode=block_global) number of blocks K; K=1 equals global; larger K makes global more local",
    )
    p.add_argument(
        "--semantic_sampler_compute_device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="device for semantic sampler computations (global/multi). Use cuda to reduce CPU bottleneck.",
    )
    p.add_argument(
        "--semantic_sampler_cache_teacher_on_device",
        action="store_true",
        help="if set and compute_device=cuda, keep teacher embedding cache on GPU for faster sampling (uses ~1-2GB VRAM)",
    )

    # Teachers (sampling only)
    p.add_argument("--cache_dir", type=str, default="./cache")
    p.add_argument("--teacher_text_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--teacher_image_arch", type=str, default="resnet50")
    p.add_argument(
        "--teacher_image_model",
        type=str,
        default=None,
        help="only for --teacher_image_arch clip; default: openai/clip-vit-base-patch32",
    )
    p.add_argument(
        "--semantic_teacher_same_as_student",
        action="store_true",
        help=(
            "Use the student's UniModalDualEncoder (vision+text+projection, eval mode) to build semantic teacher "
            "embeddings; same L2-normalized space as training features. Only for --student_arch hf_unimodal. "
            "Ignores --teacher_text_model / --teacher_image_*."
        ),
    )
    p.add_argument(
        "--prep_teacher_cache_only",
        action="store_true",
        help="Build teacher embedding cache under --cache_dir then exit (no training / eval / sampler setup).",
    )
    p.add_argument(
        "--student_teacher_image_batch_size",
        type=int,
        default=64,
        help="Batch size for StudentMirroredImageTeacher.encode_images during teacher cache build.",
    )
    p.add_argument(
        "--student_teacher_text_batch_size",
        type=int,
        default=128,
        help="Batch size for StudentMirroredTextTeacher.encode during teacher cache build.",
    )

    # Semantic relation definition
    # Keep defaults consistent with SemanticRelationConfig
    p.add_argument("--text_sim_threshold", type=float, default=0.55)
    p.add_argument("--image_sim_threshold", type=float, default=0.70)
    p.add_argument("--use_image_topk", action="store_true")
    p.add_argument("--image_topk", type=int, default=50)

    # Eval
    p.add_argument("--eval_every", type=int, default=200)
    p.add_argument("--eval_max_pairs", type=int, default=5000)
    p.add_argument(
        "--eval_type",
        type=str,
        default="retrieval",
        choices=["retrieval", "four_afc", "both"],
        help="retrieval=COCO-pairs R@K (default); four_afc=object-category 4-AFC; both=run each eval_every.",
    )
    p.add_argument(
        "--four_afc_subtype",
        type=str,
        default="image",
        choices=["image", "text"],
        help="4-AFC layout (matches mcl eval_type): image=4 images vs 1 word; text=1 image vs 4 words.",
    )
    p.add_argument(
        "--four_afc_metadata",
        type=str,
        default=None,
        help=f"JSON with key 'data' listing trials. Default: {default_four_afc_metadata_path()}",
    )
    p.add_argument(
        "--four_afc_image_root",
        type=str,
        default=None,
        help="Optional directory prepended when trial image paths are not absolute / not found.",
    )
    p.add_argument(
        "--four_afc_max_trials",
        type=int,
        default=None,
        help="If set, only evaluate the first N trials (faster smoke tests).",
    )

    # Output
    p.add_argument("--run_dir", type=str, default="./runs/dev")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    seed_all(args.seed)
    device = get_device(args.device)
    if device.type == "cuda" and not bool(args.deterministic):
        torch.backends.cudnn.benchmark = True

    if args.deterministic:
        # Best-effort determinism (may not cover all ops/hardware).
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Dataset
    if args.dataset_type == "coco":
        if args.coco_images_dir is None or args.coco_captions_json is None:
            raise SystemExit("When dataset_type=coco, --coco_images_dir and --coco_captions_json are required.")
        dataset = CocoPairsDataset(
            args.coco_images_dir,
            args.coco_captions_json,
            max_pairs=args.max_pairs,
        )
    elif args.dataset_type == "cc3m_wds":
        if args.wds_root is None:
            raise SystemExit("When dataset_type=cc3m_wds, --wds_root is required.")
        dataset = WdsPairsDataset(
            args.wds_root,
            shard_glob=str(args.wds_shard_glob),
            max_pairs=args.max_pairs,
            rebuild_index=bool(args.wds_rebuild_index),
        )
    else:
        if args.saycam_images_root is None or args.saycam_metadata_json is None:
            raise SystemExit(
                "When dataset_type=saycam, --saycam_images_root and --saycam_metadata_json are required."
            )
        dataset = SayCamPairsDataset(
            args.saycam_images_root,
            args.saycam_metadata_json,
            mode=args.saycam_mode,
            max_pairs=args.max_pairs,
            seed=args.seed,
        )

    # Main model + processors
    image_processor = None
    tokenizer = None
    if str(args.student_arch) == "cvcl":
        # Lazy import: avoid requiring CVCL dependencies when using the default HF student.
        from samcl.models.cvcl_dual_encoder import CvclDualEncoder, CvclDualEncoderConfig

        vocab = build_cvcl_vocab_from_dataset(
            dataset,
            min_freq=int(args.cvcl_vocab_min_freq),
            max_vocab=int(args.cvcl_vocab_max),
        )
        collate_fn = CvclCollator(
            vocab=vocab,
            image_size=int(args.cvcl_image_size),
            use_strong_aug=bool(args.cvcl_use_strong_aug),
        )
        model = CvclDualEncoder(
            CvclDualEncoderConfig(
                vision_model_name=str(args.cvcl_vision_model_name),
                vision_pretrained=(not bool(args.cvcl_vision_from_scratch)),
                vocab_size=len(vocab.word2idx),
                embed_dim=512,
                init_temperature=float(args.init_temperature),
            )
        ).to(device)
    elif str(args.student_arch) == "cvcl_vision_hf_text":
        # CVCL vision backbone + HF text backbone (original language encoder family).
        from samcl.models.hybrid_cvclvision_hftext import (
            HybridCvclVisionHfText,
            HybridCvclVisionHfTextConfig,
            load_hf_text_tokenizer,
        )

        tokenizer = load_hf_text_tokenizer(text_model_name=str(args.text_model_name))
        collate_fn = CvclCollator(
            # Reuse CVCL image pipeline; override text path in collator below.
            vocab=build_cvcl_vocab_from_dataset(dataset, min_freq=1, max_vocab=2),  # placeholder (unused)
            image_size=int(args.cvcl_image_size),
            use_strong_aug=bool(args.cvcl_use_strong_aug),
        )
        # Patch: replace collator's text processing with HF tokenizer
        from samcl.data.collate import CvclImageHfTextCollator

        collate_fn = CvclImageHfTextCollator(
            text_tokenizer=tokenizer,
            max_length=int(args.max_text_len),
            image_size=int(args.cvcl_image_size),
            use_strong_aug=bool(args.cvcl_use_strong_aug),
        )
        model = HybridCvclVisionHfText(
            HybridCvclVisionHfTextConfig(
                cvcl_vision_model_name=str(args.cvcl_vision_model_name),
                cvcl_vision_pretrained=(not bool(args.cvcl_vision_from_scratch)),
                text_model_name=str(args.text_model_name),
                text_pretrained=(not bool(args.text_from_scratch)),
                proj_dim=int(args.proj_dim),
                proj_hidden_dim=int(args.proj_hidden_dim),
                proj_layers=int(args.proj_layers),
                init_temperature=float(args.init_temperature),
            )
        ).to(device)
    else:
        image_processor, tokenizer = load_unimodal_processors(
            vision_model_name=args.vision_model_name,
            text_model_name=args.text_model_name,
        )
        collate_fn = ClipCollator(
            image_processor=image_processor, text_tokenizer=tokenizer, max_length=args.max_text_len
        )
        model = UniModalDualEncoder(
            UniModalDualEncoderConfig(
                vision_model_name=args.vision_model_name,
                text_model_name=args.text_model_name,
                vision_pretrained=(not bool(args.vision_from_scratch)),
                text_pretrained=(not bool(args.text_from_scratch)),
                proj_dim=int(args.proj_dim),
                proj_hidden_dim=int(args.proj_hidden_dim),
                proj_layers=int(args.proj_layers),
                init_temperature=float(args.init_temperature),
            )
        ).to(device)

    if str(args.pairwise_loss) == "siglip":
        if str(args.student_arch) != "hf_unimodal":
            raise SystemExit("--pairwise_loss siglip is only supported for --student_arch hf_unimodal (UniModalDualEncoder).")
        # SigLIP defaults (paper): t' = log(10), b = -10; overrides --init_temperature-derived logit_scale init.
        with torch.no_grad():
            model.logit_scale.fill_(math.log(10.0))
            model.logit_bias.fill_(-10.0)

    eval_run_retrieval = str(args.eval_type) in ("retrieval", "both")
    eval_run_four_afc = str(args.eval_type) in ("four_afc", "both")
    four_afc_meta: Path | None = None
    if eval_run_four_afc:
        if image_processor is None or tokenizer is None:
            raise SystemExit(
                "--eval_type four_afc|both requires --student_arch hf_unimodal (HF image processor + tokenizer)."
            )
        four_afc_meta = Path(str(args.four_afc_metadata)) if args.four_afc_metadata else default_four_afc_metadata_path()
        if not four_afc_meta.is_file():
            raise SystemExit(f"--eval_type includes four_afc but metadata file not found: {four_afc_meta}")
    four_afc_image_root = Path(str(args.four_afc_image_root)) if args.four_afc_image_root else None

    # Teachers + cache (sampling only)
    if bool(args.semantic_teacher_same_as_student):
        if str(args.student_arch) != "hf_unimodal":
            raise SystemExit("--semantic_teacher_same_as_student requires --student_arch hf_unimodal.")
        if image_processor is None or tokenizer is None:
            raise SystemExit("--semantic_teacher_same_as_student requires HF image processor and tokenizer.")
        if not isinstance(model, UniModalDualEncoder):
            raise SystemExit("--semantic_teacher_same_as_student requires UniModalDualEncoder.")
        tag = student_teacher_cache_tag(args)
        text_teacher = StudentMirroredTextTeacher(
            model,
            tokenizer,
            device,
            max_text_len=int(args.max_text_len),
            batch_size=int(args.student_teacher_text_batch_size),
        )
        image_teacher = StudentMirroredImageTeacher(
            model, image_processor, device, batch_size=int(args.student_teacher_image_batch_size)
        )
        teacher_cache = TeacherEmbeddingCache(
            args.cache_dir,
            dataset=dataset,
            text_teacher=text_teacher,
            image_teacher=image_teacher,
            expected_teacher_tag=tag,
        )
        was_training = model.training
        model.eval()
        try:
            teacher_cache.ensure_built()
        finally:
            if was_training:
                model.train()
    else:
        text_teacher = FrozenTextTeacher(TextTeacherConfig(model_name=args.teacher_text_model), device=device)
        image_teacher = FrozenImageTeacher(
            ImageTeacherConfig(
                arch=args.teacher_image_arch,
                clip_model_name=(args.teacher_image_model or "openai/clip-vit-base-patch32"),
            ),
            device=device,
        )
        teacher_cache = TeacherEmbeddingCache(
            args.cache_dir,
            dataset=dataset,
            text_teacher=text_teacher,
            image_teacher=image_teacher,
            expected_teacher_tag=None,
        )
        teacher_cache.ensure_built()

    if bool(args.prep_teacher_cache_only):
        append_jsonl(
            metrics_path,
            {
                "type": "prep_teacher_cache_only_done",
                "cache_dir": str(Path(args.cache_dir).resolve()),
                "num_pairs": int(len(dataset)),
                "semantic_teacher_same_as_student": bool(args.semantic_teacher_same_as_student),
            },
        )
        return

    oracle = SemanticRelationOracle(
        dataset=dataset,
        teacher_cache=teacher_cache,
        cfg=SemanticRelationConfig(
            text_sim_threshold=args.text_sim_threshold,
            image_sim_threshold=args.image_sim_threshold,
            use_image_topk=bool(args.use_image_topk),
            image_topk=int(args.image_topk),
            relation_mode=str(args.semantic_relation_mode),
        ),
    )

    # Sampler (only thing changing across experiments)
    if args.sampling_strategy == "random":
        batch_sampler = RandomBatchSampler(dataset, batch_size=args.batch_size, drop_last=True, seed=args.seed)
    else:
        rel_mode = str(args.semantic_relation_mode).lower().strip()
        if rel_mode == "full":
            mix = SemanticMix(args.mix_r1, args.mix_r2, args.mix_r3, args.mix_r4)
        else:
            mix = BinaryMix(similar=float(args.mix_similar), different=float(args.mix_different))
        batch_sampler = SemanticBatchSampler(
            dataset,
            oracle=oracle,
            mix=mix,
            relation_mode=rel_mode,
            mode=str(args.semantic_sampler_mode),
            num_anchors=int(args.semantic_sampler_num_anchors),
            min_anchor_matches=args.semantic_sampler_min_anchor_matches,
            global_num_candidates=int(args.semantic_sampler_global_num_candidates),
            num_blocks=int(args.semantic_sampler_num_blocks),
            compute_device=str(args.semantic_sampler_compute_device),
            cache_teacher_on_device=bool(args.semantic_sampler_cache_teacher_on_device),
            batch_size=args.batch_size,
            drop_last=True,
            seed=args.seed,
            max_tries=int(args.sampler_max_tries),
        )

    def _make_worker_init(worker_threads: int):
        def _seed_worker(worker_id: int) -> None:
            wt = int(worker_threads)
            if wt > 0:
                try:
                    torch.set_num_threads(wt)
                except Exception:
                    pass
            # Make numpy/random deterministic per worker.
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            import random as _random

            _random.seed(worker_seed)

        return _seed_worker

    generator = torch.Generator()
    generator.manual_seed(int(args.seed))

    nw = int(args.num_workers)
    dl_kwargs = {
        "dataset": dataset,
        "batch_sampler": batch_sampler,
        "num_workers": nw,
        "pin_memory": bool(args.pin_memory),
        "collate_fn": collate_fn,
        "worker_init_fn": _make_worker_init(int(args.dataloader_worker_threads)) if nw > 0 else None,
        "generator": generator,
        "persistent_workers": nw > 0,
    }
    if nw > 0:
        pf = int(args.prefetch_factor)
        if pf < 2:
            raise SystemExit("--prefetch_factor must be >= 2 when num_workers > 0 (PyTorch requirement).")
        dl_kwargs["prefetch_factor"] = pf
    loader = DataLoader(**dl_kwargs)

    opt = AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    burner = None
    if int(args.gpu_burn_every) > 0 and device.type == "cuda":
        burner = GpuMatmulBurner(
            device=device,
            matmul_dim=int(args.gpu_burn_matmul_dim),
            iters=int(args.gpu_burn_iters),
            dtype=str(args.gpu_burn_dtype),
            reserve_gb=float(args.gpu_burn_reserve_gb),
        )

    def _save_checkpoint(*, path: Path, step: int, samples_seen: int, epoch: int, batch_in_epoch: int) -> None:
        payload = {
            "step": int(step),
            "samples_seen": int(samples_seen),
            "epoch": int(epoch),
            "batch_in_epoch": int(batch_in_epoch),
            "args": vars(args),
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "torch_cuda": (torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
            },
        }
        torch.save(payload, path)

    def _maybe_load_checkpoint(path: Path) -> tuple[int, int, int, int]:
        ckpt = torch.load(path, map_location=device)
        incompat = model.load_state_dict(ckpt["model"], strict=False)
        if incompat.missing_keys or incompat.unexpected_keys:
            tqdm.write(
                f"[ckpt] model.load_state_dict(strict=False): missing={incompat.missing_keys}, unexpected={incompat.unexpected_keys}"
            )
        opt.load_state_dict(ckpt["optimizer"])

        # Restore RNG states best-effort
        rng = ckpt.get("rng", {})
        try:
            random.setstate(rng.get("python"))
        except Exception:
            pass
        try:
            np.random.set_state(rng.get("numpy"))
        except Exception:
            pass
        try:
            torch.set_rng_state(rng.get("torch"))
        except Exception:
            pass
        try:
            cuda_states = rng.get("torch_cuda")
            if cuda_states is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(cuda_states)
        except Exception:
            pass

        return (
            int(ckpt.get("step", 0)),
            int(ckpt.get("samples_seen", 0)),
            int(ckpt.get("epoch", 0)),
            int(ckpt.get("batch_in_epoch", 0)),
        )

    # Save run config
    append_jsonl(metrics_path, {"type": "config", "args": vars(args)})

    step = 0
    samples_seen = 0
    model.train()

    train_start_t = time.perf_counter()
    pbar = tqdm(total=args.max_steps, desc="train_steps")
    epoch = 0
    batch_in_epoch = 0

    # Resume logic
    resume_path: Path | None = None
    if args.resume_from:
        resume_path = Path(str(args.resume_from))
    elif bool(args.auto_resume):
        candidate = ckpt_dir / "latest.pt"
        if candidate.exists():
            resume_path = candidate

    ckpt_batch_in_epoch = 0
    if resume_path is not None and resume_path.exists():
        step, samples_seen, epoch, ckpt_batch_in_epoch = _maybe_load_checkpoint(resume_path)
        # IMPORTANT: Strictly resuming the dataloader position requires skipping batches,
        # which is prohibitively slow for expensive samplers (e.g., semantic global mode).
        # By default we resume model/optimizer/step, but restart the epoch iteration.
        batch_in_epoch = int(ckpt_batch_in_epoch) if bool(args.resume_strict) else 0
        pbar.update(int(step))
        append_jsonl(
            metrics_path,
            {
                "type": "resume",
                "from": str(resume_path),
                "step": int(step),
                "samples_seen": int(samples_seen),
                "epoch": int(epoch),
                "ckpt_batch_in_epoch": int(ckpt_batch_in_epoch),
                "resume_strict": bool(args.resume_strict),
                "batch_in_epoch": int(batch_in_epoch),
            },
        )

    # GPU memory tracking (per-step peak). Used to report average usage excluding peak.
    gpu_peak_alloc_mb: list[float] = []
    gpu_peak_reserved_mb: list[float] = []
    while step < args.max_steps:
        if hasattr(batch_sampler, "set_epoch"):
            batch_sampler.set_epoch(epoch)
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)
        epoch += 1
        cur_epoch_batch_idx = 0
        for batch in loader:
            if step >= args.max_steps:
                break
            # If strict resume is enabled, skip already-consumed batches within the epoch.
            # Note: this can be extremely slow for expensive batch samplers.
            if bool(args.resume_strict) and batch_in_epoch > 0 and cur_epoch_batch_idx < batch_in_epoch:
                cur_epoch_batch_idx += 1
                continue

            step_start_t = time.perf_counter()
            if device.type == "cuda":
                try:
                    torch.cuda.reset_peak_memory_stats(device)
                except Exception:
                    # Some backends/devices may not support this; best-effort.
                    pass
            non_block = bool(args.pin_memory) and device.type == "cuda"
            pixel_values = batch["pixel_values"].to(device, non_blocking=non_block)
            input_ids = batch["input_ids"].to(device, non_blocking=non_block)
            attention_mask = (
                batch["attention_mask"].to(device, non_blocking=non_block)
                if batch["attention_mask"] is not None
                else None
            )

            out = model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask)
            if str(args.pairwise_loss) == "siglip":
                loss = siglip_pairwise_loss(out["logits_per_image"])
            else:
                loss = clip_infonce_loss(out["logits_per_image"], out["logits_per_text"])

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            step += 1
            bsz = int(pixel_values.shape[0])
            samples_seen += bsz
            cur_epoch_batch_idx += 1
            batch_in_epoch = cur_epoch_batch_idx
            step_time_s = time.perf_counter() - step_start_t
            elapsed_s = time.perf_counter() - train_start_t

            # Optional: burn extra GPU compute to increase utilization.
            if burner is not None and int(args.gpu_burn_every) > 0 and (step % int(args.gpu_burn_every) == 0):
                burner()

            # Track per-step GPU peak memory (MB) if available.
            if device.type == "cuda":
                try:
                    gpu_peak_alloc_mb.append(float(torch.cuda.max_memory_allocated(device) / (1024**2)))
                    gpu_peak_reserved_mb.append(float(torch.cuda.max_memory_reserved(device) / (1024**2)))
                except Exception:
                    pass

            # At step 10, report average GPU usage excluding the single highest peak.
            if step == 10 and device.type == "cuda" and len(gpu_peak_alloc_mb) >= 2:
                alloc_vals = gpu_peak_alloc_mb[:10]
                resv_vals = gpu_peak_reserved_mb[:10] if len(gpu_peak_reserved_mb) >= 10 else gpu_peak_reserved_mb
                if len(alloc_vals) >= 2:
                    alloc_peak = max(alloc_vals)
                    alloc_avg_excl_peak = (sum(alloc_vals) - alloc_peak) / (len(alloc_vals) - 1)
                else:
                    alloc_peak = float("nan")
                    alloc_avg_excl_peak = float("nan")

                if len(resv_vals) >= 2:
                    resv_peak = max(resv_vals)
                    resv_avg_excl_peak = (sum(resv_vals) - resv_peak) / (len(resv_vals) - 1)
                else:
                    resv_peak = float("nan")
                    resv_avg_excl_peak = float("nan")

                msg = (
                    "[gpu_mem@step10] "
                    f"avg_alloc_excl_peak={alloc_avg_excl_peak:.1f}MB (peak={alloc_peak:.1f}MB), "
                    f"avg_reserved_excl_peak={resv_avg_excl_peak:.1f}MB (peak={resv_peak:.1f}MB)"
                )
                tqdm.write(msg)
                append_jsonl(
                    metrics_path,
                    {
                        "type": "gpu_mem",
                        "step": step,
                        "samples_seen": samples_seen,
                        "avg_alloc_mb_excl_peak": float(alloc_avg_excl_peak),
                        "peak_alloc_mb": float(alloc_peak),
                        "avg_reserved_mb_excl_peak": float(resv_avg_excl_peak),
                        "peak_reserved_mb": float(resv_peak),
                        "time_elapsed_s": float(elapsed_s),
                    },
                )

            # Sampler diagnostics (per-step, lightweight).
            # For semantic sampler, note: the first element of each batch is always a random seed pair.
            sampler_hit = None
            sampler_fallback = None
            sampler_avg_tries = None
            if batch.get("sample_is_fallback", None) is not None and batch.get("sample_tries", None) is not None:
                is_fb = batch["sample_is_fallback"]
                tries = batch["sample_tries"]
                sampler_fallback = int(is_fb.sum().item())
                sampler_hit = int(is_fb.numel() - sampler_fallback)
                # Average tries over samples that have a target relation (i.e., exclude random sampler and seed item).
                if batch.get("sample_target_relation", None) is not None:
                    tgt = batch["sample_target_relation"]
                    # Exclude the seed item (target_relation=-1). For global mode we use -2 and want to include it.
                    mask = tgt.ne(-1)
                    if int(mask.sum().item()) > 0:
                        sampler_avg_tries = float(tries[mask].float().mean().item())
                    else:
                        sampler_avg_tries = 0.0
                else:
                    sampler_avg_tries = float(tries.float().mean().item())

                # To keep logs compact, write sampler diagnostics every 10 steps.
                if step == 1 or (step % 10 == 0):
                    append_jsonl(
                        metrics_path,
                        {
                            "type": "sampler",
                            "step": step,
                            "samples_seen": samples_seen,
                            "sampler_hit": sampler_hit,
                            "sampler_fallback": sampler_fallback,
                            "sampler_avg_tries": sampler_avg_tries,
                            "time_elapsed_s": float(elapsed_s),
                            "step_time_s": float(step_time_s),
                        },
                    )
            pbar.update(1)
            if step_time_s > 0:
                pbar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "s/step": f"{step_time_s:.3f}",
                        "samples/s": f"{(bsz / step_time_s):.1f}",
                        "hit": sampler_hit if sampler_hit is not None else "-",
                        "fb": sampler_fallback if sampler_fallback is not None else "-",
                        "tries": f"{sampler_avg_tries:.1f}" if sampler_avg_tries is not None else "-",
                    }
                )

            if args.log_every > 0 and (step % args.log_every == 0 or step == 1):
                with torch.no_grad():
                    rel_mode = str(args.semantic_relation_mode).lower().strip()
                    hist = oracle.batch_cross_relation_histogram_mode(
                        batch["image_id"].cpu(), batch["caption_id"].cpu(), mode=rel_mode
                    )
                    denom = max(1, int(pixel_values.shape[0]) * (int(pixel_values.shape[0]) - 1))
                    hist_pct = {k: v / denom for k, v in hist.items()}
                    full_hist = None
                    full_hist_pct = None
                    if rel_mode != "full":
                        full_hist = oracle.batch_cross_relation_histogram_mode(
                            batch["image_id"].cpu(), batch["caption_id"].cpu(), mode="full"
                        )
                        full_hist_pct = {k: v / denom for k, v in full_hist.items()}

                append_jsonl(
                    metrics_path,
                    {
                        "type": "train",
                        "step": step,
                        "samples_seen": samples_seen,
                        "loss": float(loss.item()),
                        "logit_scale": float(out["logit_scale"].detach().cpu().item()),
                        "batch_rel_counts": hist,
                        "batch_rel_pct": hist_pct,
                        "batch_rel_full_counts": full_hist,
                        "batch_rel_full_pct": full_hist_pct,
                        "sampler_hit": sampler_hit,
                        "sampler_fallback": sampler_fallback,
                        "sampler_avg_tries": sampler_avg_tries,
                        "time_elapsed_s": float(elapsed_s),
                        "step_time_s": float(step_time_s),
                        "throughput_samples_per_s": float(bsz / max(1e-9, step_time_s)),
                    },
                )

            if args.eval_every > 0 and (step % args.eval_every == 0):
                model.eval()
                if eval_run_retrieval:
                    metrics = evaluate_retrieval(
                        model=model,
                        dataset=dataset,
                        collate_fn=collate_fn,
                        device=device,
                        batch_size=args.batch_size,
                        max_pairs=args.eval_max_pairs,
                    )
                    tqdm.write(
                        "eval "
                        f"step={step} samples_seen={samples_seen} "
                        f"I2T R@1/5/10={metrics.i2t_r1:.3f}/{metrics.i2t_r5:.3f}/{metrics.i2t_r10:.3f} "
                        f"T2I R@1/5/10={metrics.t2i_r1:.3f}/{metrics.t2i_r5:.3f}/{metrics.t2i_r10:.3f}"
                    )
                    append_jsonl(metrics_path, {"type": "eval", "step": step, "samples_seen": samples_seen, **asdict(metrics)})
                if eval_run_four_afc:
                    assert four_afc_meta is not None
                    n_cap = args.four_afc_max_trials
                    n_show = int(n_cap) if n_cap is not None else 5000
                    m4 = evaluate_four_afc(
                        model=model,
                        image_processor=image_processor,
                        tokenizer=tokenizer,
                        metadata_path=four_afc_meta,
                        device=device,
                        subtype=str(args.four_afc_subtype),
                        max_text_len=int(args.max_text_len),
                        max_trials=int(n_cap) if n_cap is not None else None,
                        image_root=four_afc_image_root,
                        show_progress=bool(n_show > 256),
                    )
                    tqdm.write(
                        f"eval_four_afc step={step} samples_seen={samples_seen} "
                        f"subtype={m4.subtype} acc={m4.accuracy:.4f} n={m4.n_trials}"
                    )
                    append_jsonl(
                        metrics_path,
                        {
                            "type": "eval_four_afc",
                            "step": step,
                            "samples_seen": samples_seen,
                            "four_afc_accuracy": m4.accuracy,
                            "four_afc_n_trials": m4.n_trials,
                            "four_afc_subtype": m4.subtype,
                        },
                    )
                model.train()

            # Save checkpoints
            if int(args.save_every) > 0 and (step % int(args.save_every) == 0):
                if not bool(args.save_latest_only):
                    _save_checkpoint(
                        path=(ckpt_dir / f"step_{step}.pt"),
                        step=step,
                        samples_seen=samples_seen,
                        epoch=epoch,
                        batch_in_epoch=batch_in_epoch,
                    )
                _save_checkpoint(
                    path=(ckpt_dir / "latest.pt"),
                    step=step,
                    samples_seen=samples_seen,
                    epoch=epoch,
                    batch_in_epoch=batch_in_epoch,
                )
                append_jsonl(
                    metrics_path,
                    {
                        "type": "checkpoint",
                        "step": int(step),
                        "samples_seen": int(samples_seen),
                        "path": str(ckpt_dir / "latest.pt"),
                    },
                )

    pbar.close()
    total_train_s = time.perf_counter() - train_start_t
    # Always save a final checkpoint
    _save_checkpoint(
        path=(ckpt_dir / "latest.pt"),
        step=step,
        samples_seen=samples_seen,
        epoch=epoch,
        batch_in_epoch=batch_in_epoch,
    )
    append_jsonl(
        metrics_path,
        {
            "type": "train_done",
            "step": step,
            "samples_seen": samples_seen,
            "total_train_time_s": float(total_train_s),
        },
    )


if __name__ == "__main__":
    main()

