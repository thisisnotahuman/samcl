import argparse
import ast
import sys


def get_default_params(model_name):
    # Params from paper (https://arxiv.org/pdf/2103.00020.pdf)
    model_name = model_name.lower()
    if "vit" in model_name:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.98, "eps": 1.0e-6}
    else:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.999, "eps": 1.0e-8}


class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        kw = {}
        for value in values:
            key, value = value.split('=')
            try:
                kw[key] = ast.literal_eval(value)
            except ValueError:
                kw[key] = str(value)  # fallback to string (avoid need to escape on command line)
        setattr(namespace, self.dest, kw)


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-data",
        type=str,
        default=None,
        help="Path to file(s) with training data. When using webdataset, multiple datasources can be combined using the `::` separator.",
    )
    parser.add_argument(
        "--train-data-upsampling-factors",
        type=str,
        default=None,
        help=(
            "When using multiple data sources with webdataset and sampling with replacement, this can be used to upsample specific data sources. "
            "Similar to --train-data, this should be a string with as many numbers as there are data sources, separated by `::` (e.g. 1::2::0.5) "
            "By default, datapoints are sampled uniformly regardless of the dataset sizes."
        )
    )
    parser.add_argument(
        "--val-data",
        type=str,
        default=None,
        help="Path to file(s) with validation data",
    )
    parser.add_argument(
        "--train-num-samples",
        type=int,
        default=None,
        help="Number of samples in dataset. Required for webdataset if not available in info file.",
    )
    parser.add_argument(
        "--val-num-samples",
        type=int,
        default=None,
        help="Number of samples in dataset. Useful for webdataset if not available in info file.",
    )
    parser.add_argument(
        "--dataset-type",
        choices=["webdataset", "csv", "synthetic", "auto"],
        default="auto",
        help="Which type of dataset to process."
    )
    parser.add_argument(
        "--dataset-resampled",
        default=False,
        action="store_true",
        help="Whether to use sampling with replacement for webdataset shard selection."
    )
    parser.add_argument(
        "--csv-separator",
        type=str,
        default="\t",
        help="For csv-like datasets, which separator to use."
    )
    parser.add_argument(
        "--csv-img-key",
        type=str,
        default="filepath",
        help="For csv-like datasets, the name of the key for the image paths."
    )
    parser.add_argument(
        "--csv-caption-key",
        type=str,
        default="title",
        help="For csv-like datasets, the name of the key for the captions."
    )
    parser.add_argument(
        "--imagenet-val",
        type=str,
        default=None,
        help="Path to imagenet val set for conducting zero shot evaluation.",
    )
    parser.add_argument(
        "--imagenet-v2",
        type=str,
        default=None,
        help="Path to imagenet v2 for conducting zero shot evaluation.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Override system default cache path for model & tokenizer file downloads.",
    )
    parser.add_argument(
        "--logs",
        type=str,
        default="./logs/",
        help="Where to store tensorboard logs. Use None to avoid storing logs.",
    )
    parser.add_argument(
        "--log-local",
        action="store_true",
        default=False,
        help="log files on local master, otherwise global master only.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Optional identifier for the experiment when storing logs. Otherwise use current time.",
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of dataloader workers per GPU."
    )
    parser.add_argument(
        "--prefetch-factor",
        dest="prefetch_factor",
        type=int,
        default=2,
        help="DataLoader prefetch_factor per worker when workers>0 (PyTorch requires >=2). Try 4-8 if CPU/RAM allow.",
    )
    parser.add_argument(
        "--dataloader-worker-torch-threads",
        dest="dataloader_worker_torch_threads",
        type=int,
        default=1,
        help="torch.set_num_threads in each DataLoader worker (1 often helps when workers>1 and decode/preprocess is CPU-heavy).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size per GPU."
    )
    parser.add_argument(
        "--epochs", type=int, default=32, help="Number of epochs to train for."
    )
    parser.add_argument(
        "--epochs-cooldown", type=int, default=None,
        help="When scheduler w/ cooldown used, perform cooldown from total_epochs - cooldown_epochs onwards."
    )
    parser.add_argument("--lr", type=float, default=None, help="Learning rate.")
    parser.add_argument("--beta1", type=float, default=None, help="Adam beta 1.")
    parser.add_argument("--beta2", type=float, default=None, help="Adam beta 2.")
    parser.add_argument("--eps", type=float, default=None, help="Adam epsilon.")
    parser.add_argument("--wd", type=float, default=0.2, help="Weight decay.")
    parser.add_argument("--momentum", type=float, default=None, help="Momentum (for timm optimizers).")
    parser.add_argument(
        "--warmup", type=int, default=10000, help="Number of steps to warmup for."
    )
    parser.add_argument(
        "--opt", type=str, default='adamw',
        help="Which optimizer to use. Choices are ['adamw', or any timm optimizer 'timm/{opt_name}']."
    )
    parser.add_argument(
        "--use-bn-sync",
        default=False,
        action="store_true",
        help="Whether to use batch norm sync.")
    parser.add_argument(
        "--skip-scheduler",
        action="store_true",
        default=False,
        help="Use this flag to skip the learning rate decay.",
    )
    parser.add_argument(
        "--max-steps",
        dest="max_steps",
        type=int,
        default=None,
        help=(
            "Stop after this many optimizer steps (global), like src/samcl/train.py --max_steps. "
            "Exits the epoch loop early once reached. Independent of --epochs unless you rely on multiple epochs "
            "to reach the cap."
        ),
    )
    parser.add_argument(
        "--samcl-train-align",
        dest="samcl_train_align",
        action="store_true",
        default=False,
        help=(
            "Match legacy samcl.train defaults: --lr-scheduler const, --warmup 0, wd=0.01, lr=1e-5 "
            "(only if you did not pass --lr), max_steps=2000 unless --max-steps is set. "
            "Does not change the CLIP model or SAMCL sampling; only optimization horizon and schedule."
        ),
    )
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        default='cosine',
        help="LR scheduler. One of: 'cosine', 'const' (constant), 'const-cooldown' (constant w/ cooldown). Default: cosine",
    )
    parser.add_argument(
        "--lr-cooldown-end", type=float, default=0.0,
        help="End learning rate for cooldown schedule. Default: 0"
    )
    parser.add_argument(
        "--lr-cooldown-power", type=float, default=1.0,
        help="Power for polynomial cooldown schedule. Default: 1.0 (linear decay)"
    )
    parser.add_argument(
        "--save-frequency", type=int, default=1, help="How often to save checkpoints."
    )
    parser.add_argument(
        "--save-most-recent",
        action="store_true",
        default=False,
        help="Always save the most recent model trained to epoch_latest.pt.",
    )
    parser.add_argument(
        "--save-most-recent-every-n-steps",
        dest="save_most_recent_every_n_steps",
        type=int,
        default=0,
        help=(
            "When --save-most-recent is set, also rewrite epoch_latest.pt every N optimizer steps "
            "(0 = only at end of each epoch). Use e.g. 500–5000 for long single-epoch jobs under time limits."
        ),
    )
    parser.add_argument(
        "--zeroshot-frequency", type=int, default=2, help="How often to run zero shot."
    )
    parser.add_argument(
        "--val-frequency", type=int, default=1, help="How often to run evaluation with val data."
    )
    parser.add_argument(
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "--precision",
        choices=["amp", "amp_bf16", "amp_bfloat16", "bf16", "fp16", "pure_bf16", "pure_fp16", "fp32"],
        default="amp",
        help="Floating point precision."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="RN50",
        help="Name of the vision backbone to use.",
    )
    parser.add_argument(
        "--pretrained",
        default='',
        type=str,
        help="Use a pretrained CLIP model weights with the specified tag or file path.",
    )
    parser.add_argument(
        "--pretrained-image",
        default=False,
        action='store_true',
        help="Load imagenet pretrained weights for image tower backbone if available.",
    )
    parser.add_argument(
        "--lock-image",
        default=False,
        action='store_true',
        help="Lock full image tower by disabling gradients.",
    )
    parser.add_argument(
        "--lock-image-unlocked-groups",
        type=int,
        default=0,
        help="Leave last n image tower layer groups unlocked.",
    )
    parser.add_argument(
        "--lock-image-freeze-bn-stats",
        default=False,
        action='store_true',
        help="Freeze BatchNorm running stats in image tower for any locked layers.",
    )
    parser.add_argument(
        '--image-mean', type=float, nargs='+', default=None, metavar='MEAN',
        help='Override default image mean value of dataset')
    parser.add_argument(
        '--image-std', type=float, nargs='+', default=None, metavar='STD',
        help='Override default image std deviation of of dataset')
    parser.add_argument(
        '--image-interpolation',
        default=None, type=str, choices=['bicubic', 'bilinear', 'random'],
        help="Override default image resize interpolation"
    )
    parser.add_argument(
        '--image-resize-mode',
        default=None, type=str, choices=['shortest', 'longest', 'squash'],
        help="Override default image resize (& crop) mode during inference"
    )
    parser.add_argument('--aug-cfg', nargs='*', default={}, action=ParseKwargs)
    parser.add_argument(
        "--grad-checkpointing",
        default=False,
        action='store_true',
        help="Enable gradient checkpointing.",
    )
    parser.add_argument(
        "--local-loss",
        default=False,
        action="store_true",
        help="calculate loss w/ local features @ global (instead of realizing full global @ global matrix)"
    )
    parser.add_argument(
        "--gather-with-grad",
        default=False,
        action="store_true",
        help="enable full distributed gradient for feature gather"
    )
    parser.add_argument(
        '--force-context-length', type=int, default=None,
        help='Override default context length'
    )
    parser.add_argument(
        '--force-image-size', type=int, nargs='+', default=None,
        help='Override default image size'
    )
    parser.add_argument(
        "--force-quick-gelu",
        default=False,
        action='store_true',
        help="Force use of QuickGELU activation for non-OpenAI transformer models.",
    )
    parser.add_argument(
        "--force-patch-dropout",
        default=None,
        type=float,
        help="Override the patch dropout during training, for fine tuning with no dropout near the end as in the paper",
    )
    parser.add_argument(
        "--force-custom-text",
        default=False,
        action='store_true',
        help="Force use of CustomTextCLIP model (separate text-tower).",
    )
    parser.add_argument(
        "--torchscript",
        default=False,
        action='store_true',
        help="torch.jit.script the model, also uses jit version of OpenAI models if pretrained=='openai'",
    )
    parser.add_argument(
        "--torchcompile",
        default=False,
        action='store_true',
        help="torch.compile() the model, requires pytorch 2.0 or later.",
    )
    parser.add_argument(
        "--trace",
        default=False,
        action='store_true',
        help="torch.jit.trace the model for inference / eval only",
    )
    parser.add_argument(
        "--accum-freq", type=int, default=1, help="Update the model every --acum-freq steps."
    )
    parser.add_argument(
        "--extra-gpu-reserve-mib",
        dest="extra_gpu_reserve_mib",
        type=int,
        default=0,
        help=(
            "Reserve extra CUDA memory (MiB) and run a large matmul each training step to keep GPU "
            "utilization/memory high for strict cluster policies (0 disables). If you pass "
            "--extra-gpu-util-boost, the background booster owns these buffers and the per-step matmul "
            "is disabled to avoid concurrent use. May slow training."
        ),
    )
    parser.add_argument(
        "--extra-gpu-util-boost",
        dest="extra_gpu_util_boost",
        default=False,
        action="store_true",
        help=(
            "Run continuous extra GPU matmuls in a background thread to keep utilization high even "
            "during dataloader stalls. Use with --extra-gpu-reserve-mib. May slow training."
        ),
    )
    parser.add_argument(
        "--extra-gpu-util-boost-sleep-ms",
        dest="extra_gpu_util_boost_sleep_ms",
        type=int,
        default=0,
        help="Optional fixed sleep after each boost burst (ms). Ignored if --extra-gpu-boost-duty is set.",
    )
    parser.add_argument(
        "--extra-gpu-boost-matmul-iters",
        dest="extra_gpu_boost_matmul_iters",
        type=int,
        default=32,
        help=(
            "How many large matmuls to run per boost loop iteration (higher = higher SM util, more "
            "interference with training). Only used with --extra-gpu-util-boost."
        ),
    )
    parser.add_argument(
        "--extra-gpu-boost-duty",
        dest="extra_gpu_boost_duty",
        type=float,
        default=None,
        help=(
            "Throttle GPU util boost: after each synchronized burst, sleep so that roughly this "
            "fraction of wall time is spent in bursts (e.g. 0.52 ≈ 52%% average while dataloader is "
            "stalled). Must be in (0,1). When set, overrides --extra-gpu-util-boost-sleep-ms between bursts."
        ),
    )
    parser.add_argument(
        "--extra-gpu-boost-idle-only",
        dest="extra_gpu_boost_idle_only",
        default=False,
        action="store_true",
        help=(
            "With --extra-gpu-util-boost, run boost only while blocked waiting for the next batch "
            "(DataLoader). Pauses during forward/backward so training is faster; long I/O waits still "
            "get boosted GPU util."
        ),
    )
    parser.add_argument(
        "--log-gpu-util-window-sec",
        dest="log_gpu_util_window_sec",
        type=float,
        default=120.0,
        help="When logging Train Epoch lines, also append mean GPU util over last N seconds (0 disables).",
    )
    parser.add_argument(
        "--log-gpu-util-subsample-sec",
        dest="log_gpu_util_subsample_sec",
        type=float,
        default=5.0,
        help="How often to poll nvidia-smi for GPU util averaging (smaller = smoother but more overhead).",
    )
    parser.add_argument(
        "--device", default="cuda", type=str, help="Accelerator to use."
    )
    # arguments for distributed training
    parser.add_argument(
        "--dist-url",
        default=None,
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--dist-backend",
        default=None,
        type=str,
        help="distributed backend. \"nccl\" for GPU, \"hccl\" for Ascend NPU"
    )
    parser.add_argument(
        "--report-to",
        default='',
        type=str,
        help="Options are ['wandb', 'tensorboard', 'wandb,tensorboard']"
    )
    parser.add_argument(
        "--wandb-notes",
        default='',
        type=str,
        help="Notes if logging with wandb"
    )
    parser.add_argument(
        "--wandb-project-name",
        type=str,
        default='open-clip',
        help="Name of the project if logging with wandb.",
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="If true, more information is logged."
    )
    parser.add_argument(
        "--copy-codebase",
        default=False,
        action="store_true",
        help="If true, we copy the entire base on the log directory, and execute from there."
    )
    parser.add_argument(
        "--horovod",
        default=False,
        action="store_true",
        help="Use horovod for distributed training."
    )
    parser.add_argument(
        "--ddp-static-graph",
        default=False,
        action='store_true',
        help="Enable static graph optimization for DDP in PyTorch >= 1.11.",
    )
    parser.add_argument(
        "--no-set-device-rank",
        default=False,
        action="store_true",
        help="Don't set device index from local rank (when CUDA_VISIBLE_DEVICES restricted to one per proc)."
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Default random seed."
    )
    parser.add_argument(
        "--grad-clip-norm", type=float, default=None, help="Gradient clip."
    )
    parser.add_argument(
        "--lock-text",
        default=False,
        action='store_true',
        help="Lock full text tower by disabling gradients.",
    )
    parser.add_argument(
        "--lock-text-unlocked-layers",
        type=int,
        default=0,
        help="Leave last n text tower layer groups unlocked.",
    )
    parser.add_argument(
        "--lock-text-freeze-layer-norm",
        default=False,
        action='store_true',
        help="Freeze LayerNorm running stats in text tower for any locked layers.",
    )
    parser.add_argument(
        "--log-every-n-steps",
        type=int,
        default=100,
        help="Log every n steps to tensorboard/console/wandb.",
    )
    parser.add_argument(
        "--coca-caption-loss-weight",
        type=float,
        default=2.0,
        help="Weight assigned to caption loss in CoCa."
    )
    parser.add_argument(
        "--coca-contrastive-loss-weight",
        type=float,
        default=1.0,
        help="Weight assigned to contrastive loss when training CoCa."
    )
    parser.add_argument(
        "--remote-sync",
        type=str,
        default=None,
        help="Optinoally sync with a remote path specified by this arg",
    )
    parser.add_argument(
        "--remote-sync-frequency",
        type=int,
        default=300,
        help="How frequently to sync to a remote directly if --remote-sync is not None.",
    )
    parser.add_argument(
        "--remote-sync-protocol",
        choices=["s3", "fsspec"],
        default="s3",
        help="How to do the remote sync backup if --remote-sync is not None.",
    )
    parser.add_argument(
        "--delete-previous-checkpoint",
        default=False,
        action="store_true",
        help="If true, delete previous checkpoint after storing a new one."
    )
    parser.add_argument(
        "--distill-model",
        default=None,
        help='Which model arch to distill from, if any.'
    )
    parser.add_argument(
        "--distill-pretrained",
        default=None,
        help='Which pre-trained weights to distill from, if any.'
    )
    parser.add_argument(
        "--use-bnb-linear",
        default=None,
        help='Replace the network linear layers from the bitsandbytes library. '
        'Allows int8 training/inference, etc.'
    )
    parser.add_argument(
        "--siglip",
        default=False,
        action="store_true",
        help='Use SigLip (sigmoid) loss.'
    )
    parser.add_argument(
        "--loss-dist-impl",
        default=None,
        type=str,
        help='A string to specify a specific distributed loss implementation.'
    )

    # SAMCL-style semantic batch sampling (CSV/TSV or WebDataset shard directory; see data_semantic.py)
    parser.add_argument(
        "--semantic-sampling",
        default=False,
        action="store_true",
        help=(
            "Use SAMCL semantic batch composition. --train-data: CSV/TSV, WDS .tar directory, "
            "or COCO train images dir with --coco-captions-json."
        ),
    )
    parser.add_argument(
        "--semantic-relation-mode",
        type=str,
        choices=["full", "text_only", "image_only"],
        default="text_only",
        help="Relation labeling for cross-pairs: full 4-way or text_only/image_only 2-way.",
    )
    parser.add_argument("--mix-r1", type=float, default=0.25, help="(relation_mode=full) mix weight for relation 1.")
    parser.add_argument("--mix-r2", type=float, default=0.25, help="(relation_mode=full) mix weight for relation 2.")
    parser.add_argument("--mix-r3", type=float, default=0.25, help="(relation_mode=full) mix weight for relation 3.")
    parser.add_argument("--mix-r4", type=float, default=0.25, help="(relation_mode=full) mix weight for relation 4.")
    parser.add_argument(
        "--mix-similar",
        type=float,
        default=1.0,
        help="(relation_mode=text_only|image_only) probability mass on 'similar' relation.",
    )
    parser.add_argument(
        "--mix-different",
        type=float,
        default=0.0,
        help="(relation_mode=text_only|image_only) probability mass on 'different' relation.",
    )
    parser.add_argument(
        "--semantic-sampler-mode",
        type=str,
        default="global",
        help="Semantic sampler: global | single | multi | block_global (same as SAMCL).",
    )
    parser.add_argument("--semantic-sampler-num-anchors", type=int, default=3)
    parser.add_argument("--semantic-sampler-min-anchor-matches", type=int, default=None)
    parser.add_argument("--semantic-sampler-global-num-candidates", type=int, default=64)
    parser.add_argument("--semantic-sampler-num-blocks", type=int, default=1)
    parser.add_argument(
        "--semantic-sampler-compute-device",
        type=str,
        default="cuda",
        help="Device for teacher tensor ops inside the sampler (cuda or cpu).",
    )
    parser.add_argument(
        "--semantic-sampler-cache-teacher-on-device",
        dest="semantic_sampler_cache_teacher_on_device",
        action="store_true",
        default=True,
        help="Keep teacher embedding caches on GPU for faster sampling (default: on).",
    )
    parser.add_argument(
        "--no-semantic-sampler-cache-teacher-on-device",
        dest="semantic_sampler_cache_teacher_on_device",
        action="store_false",
        help="Disable caching teacher embeddings on GPU inside the sampler.",
    )
    parser.add_argument("--sampler-max-tries", type=int, default=3000)
    parser.add_argument(
        "--train-eval-type",
        dest="train_eval_type",
        type=str,
        default="retrieval",
        choices=["retrieval", "four_afc", "both", "none"],
        help=(
            "Periodic eval during training. "
            "retrieval = SAMCL-style COCO retrieval (R@1/5/10); "
            "four_afc = object-categories 4AFC; "
            "both = run both when configured; "
            "none = disable periodic eval hooks."
        ),
    )
    parser.add_argument(
        "--samcl-retrieval-eval-every-n-steps",
        dest="samcl_retrieval_eval_every_n_steps",
        type=int,
        default=0,
        help=(
            "Run SAMCL-style retrieval eval every N optimizer steps (0 disables). "
            "Only supported for COCO runs (requires --coco-captions-json)."
        ),
    )
    parser.add_argument(
        "--samcl-retrieval-eval-max-pairs",
        dest="samcl_retrieval_eval_max_pairs",
        type=int,
        default=5000,
        help="Max (image,caption) pairs for retrieval eval slice (0 = use all pairs).",
    )
    parser.add_argument(
        "--samcl-retrieval-eval-batch-size",
        dest="samcl_retrieval_eval_batch_size",
        type=int,
        default=128,
        help="Batch size for SAMCL retrieval eval embedding loop.",
    )
    parser.add_argument(
        "--four-afc-eval-json",
        dest="four_afc_eval_json",
        type=str,
        default=None,
        help=(
            "Path to object-categories 4AFC eval JSON (same schema as mcl eval_object_categories_*.json). "
            "Unset disables periodic 4AFC eval."
        ),
    )
    parser.add_argument(
        "--four-afc-eval-every-n-steps",
        dest="four_afc_eval_every_n_steps",
        type=int,
        default=200,
        help="Run 4AFC eval every N optimizer steps (0 disables). Requires --four-afc-eval-json.",
    )
    parser.add_argument(
        "--four-afc-eval-type",
        dest="four_afc_eval_type",
        type=str,
        default="text",
        choices=["text", "image"],
        help="4AFC task: text = 1 image vs 4 captions; image = 4 images vs 1 caption (target at index 0).",
    )
    parser.add_argument(
        "--four-afc-eval-max-trials",
        dest="four_afc_eval_max_trials",
        type=int,
        default=512,
        help="Max trials per eval call (random subset when dataset is larger). 0 = use all trials.",
    )
    parser.add_argument(
        "--teacher-text-model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer checkpoint for text teacher embeddings.",
    )
    parser.add_argument(
        "--teacher-image-arch",
        type=str,
        default="resnet50",
        choices=["resnet50", "clip"],
        help="Vision teacher: torchvision ResNet50 or HuggingFace CLIP vision tower.",
    )
    parser.add_argument(
        "--teacher-image-model",
        type=str,
        default="openai/clip-vit-base-patch32",
        help="Used when --teacher-image-arch=clip.",
    )
    parser.add_argument("--teacher-text-batch-size", type=int, default=128)
    parser.add_argument("--teacher-image-batch-size", type=int, default=64)
    parser.add_argument("--text-sim-threshold", type=float, default=0.30)
    parser.add_argument("--image-sim-threshold", type=float, default=0.65)
    parser.add_argument(
        "--use-image-topk",
        default=False,
        action="store_true",
        help="Use image top-k neighbor membership instead of threshold (slower).",
    )
    parser.add_argument("--image-topk", type=int, default=50)
    parser.add_argument(
        "--teacher-cache-dir",
        type=str,
        default=None,
        help="Optional dedicated directory for teacher embedding cache (default: under --cache-dir).",
    )
    parser.add_argument(
        "--semantic-max-pairs",
        type=int,
        default=None,
        help="Debug: cap number of pairs loaded from CSV or WDS index.",
    )
    parser.add_argument(
        "--semantic-wds-shard-glob",
        type=str,
        default="cc3m-train-*.tar",
        help="When --train-data is a directory: glob for training shards under that directory (default: cc3m-train-*.tar).",
    )
    parser.add_argument(
        "--semantic-wds-rebuild-index",
        default=False,
        action="store_true",
        help="Rebuild the cached WDS pair index under <train-data>/.openclip_semantic_index/ (e.g. after adding shards).",
    )
    parser.add_argument(
        "--coco-captions-json",
        type=str,
        default=None,
        help=(
            "COCO captions annotation JSON (e.g. captions_train2017.json). When set with --semantic-sampling, "
            "loads SAMCL-style COCO pairs; image directory is --coco-images-dir or --train-data."
        ),
    )
    parser.add_argument(
        "--coco-random-baseline-dataloader",
        default=False,
        action="store_true",
        help=(
            "Debug: load COCO pairs like --semantic-sampling + --coco-captions-json, but use uniform random batches "
            "(no teachers, no semantic BatchSampler). Do not pass --semantic-sampling."
        ),
    )
    parser.add_argument(
        "--log-grad-norm-every-n-steps",
        dest="log_grad_norm_every_n_steps",
        type=int,
        default=0,
        help=(
            "If > 0, master logs global L2 grad norm (unscaled grads under AMP) every N optimizer steps. "
            "For debugging; adds one unscale_ path when combined with grad clipping."
        ),
    )
    parser.add_argument(
        "--coco-images-dir",
        type=str,
        default=None,
        help="COCO train images root (e.g. train2017/). If unset, --train-data is used as the image directory in COCO mode.",
    )
    parser.add_argument(
        "--prep-gpu-util-boost",
        dest="prep_gpu_util_boost",
        default=False,
        action="store_true",
        help=(
            "Prep only: spawn a side process that runs large GEMMs on the same GPU to keep "
            "utilization high (e.g., for cluster policies). May slow actual teacher encoding; "
            "use with --device cuda."
        ),
    )
    parser.add_argument(
        "--prep-gpu-util-boost-size",
        dest="prep_gpu_util_boost_size",
        type=int,
        default=4096,
        help="Side matrix dim for --prep-gpu-util-boost (default 4096). Larger = heavier load.",
    )
    parser.add_argument(
        "--prep-gpu-log-util-interval",
        dest="prep_gpu_log_util_interval",
        type=float,
        default=120.0,
        help=(
            "Prep only (CUDA): every N seconds log GPU stats averaged over that window "
            "(see --prep-gpu-log-util-subsample-sec); 0 disables."
        ),
    )
    parser.add_argument(
        "--prep-gpu-log-util-subsample-sec",
        dest="prep_gpu_log_util_subsample_sec",
        type=float,
        default=5.0,
        help=(
            "Prep only: how often to poll nvidia-smi inside each log window (default 5s). "
            "Smaller = smoother mean SM%% but more subprocess/fork overhead and tokenizer warnings."
        ),
    )
    parser.add_argument(
        "--prep-debug-timings",
        dest="prep_debug_timings",
        default=False,
        action="store_true",
        help="Prep / teacher-cache build: log phase timings (dataset, text embed, WDS decode vs GPU encode, etc.).",
    )
    parser.add_argument(
        "--prep-wds-decode-workers",
        dest="prep_wds_decode_workers",
        type=int,
        default=0,
        help=(
            "WDS teacher image cache: thread pool size for JPEG decode after sequential tar read "
            "(0 = single-threaded). Try 4–8 when decode dominates; use fewer if CPU- or RAM-bound."
        ),
    )

    argv_for_explicit = sys.argv[1:] if args is None else args
    explicit_lr_in_argv = "--lr" in argv_for_explicit

    args = parser.parse_args(args)

    if 'timm' not in args.opt:
        # set default opt params based on model name (only if timm optimizer not used)
        default_params = get_default_params(args.model)
        for name, val in default_params.items():
            if getattr(args, name) is None:
                setattr(args, name, val)

    if getattr(args, "samcl_train_align", False):
        args.lr_scheduler = "const"
        args.warmup = 0
        args.wd = 0.01
        if args.max_steps is None:
            args.max_steps = 2000
        if not explicit_lr_in_argv:
            args.lr = 1e-5

    return args
