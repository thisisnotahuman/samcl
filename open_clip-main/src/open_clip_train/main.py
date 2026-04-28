import copy
import glob
import logging
import math
import os
import re
import subprocess
import sys
import random
from datetime import datetime
from functools import partial

import numpy as np
import torch
from torch import optim

try:
    import wandb
except ImportError:
    wandb = None

try:
    import torch.utils.tensorboard as tensorboard
except ImportError:
    tensorboard = None

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

from open_clip import create_model_and_transforms, trace_model, get_tokenizer, create_loss
from open_clip_train.data import get_data
from open_clip_train.distributed import is_master, init_distributed_device, broadcast_object
from open_clip_train.logger import setup_logging
from open_clip_train.params import parse_args
from open_clip_train.scheduler import cosine_lr, const_lr, const_lr_cooldown
from open_clip_train.train import train_one_epoch, evaluate
from open_clip_train.file_utils import pt_load, check_exists, start_sync_process, remote_sync


LATEST_CHECKPOINT_NAME = "epoch_latest.pt"


def _lr_batches_per_epoch(args, dataloader):
    """Batches per epoch used for LR total_steps (matches WebDataset when --train-num-samples is set)."""
    nb_data = max(1, int(dataloader.num_batches))
    nb = nb_data
    if getattr(args, "train_num_samples", None) is not None:
        gbs = int(args.batch_size) * int(args.world_size)
        nb_sched = max(1, math.ceil(int(args.train_num_samples) / gbs))
        if nb_sched > nb_data:
            logging.info(
                "LR schedule: --train-num-samples=%s implies %d batches/epoch for cosine; "
                "DataLoader has %d batches/epoch (semantic/COCO uses full pass per epoch). "
                "Horizon is max(data, train-num-samples) so resume + large global_step stays consistent.",
                args.train_num_samples,
                nb_sched,
                nb_data,
            )
            nb = nb_sched
    return nb


def _compute_scheduler_total_steps(args, dataloader, resume_scheduler_total_steps=None):
    """Optimizer steps across training for LR schedulers; restored from checkpoint when present.

    Returns (total_steps, nb_batches_per_epoch_used_for_lr) for cooldown math.
    """
    nb = _lr_batches_per_epoch(args, dataloader)
    acc = max(1, int(args.accum_freq))
    computed = (nb // acc) * int(args.epochs)
    if resume_scheduler_total_steps is not None:
        ts = int(resume_scheduler_total_steps)
        logging.info(
            "Restoring scheduler_total_steps=%s from checkpoint (fresh config would give %s).",
            ts,
            computed,
        )
        return ts, nb
    return computed, nb


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


def get_latest_checkpoint(path: str, remote : bool):
    # as writen, this glob recurses, so can pick up checkpoints across multiple sub-folders
    if remote:
        result = subprocess.run(["aws", "s3", "ls", path + "/"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(result)
        if result.returncode == 1:
            return None
        checkpoints = [os.path.join(path, x.split(' ')[-1]) for x in result.stdout.decode().split('\n')[:-1]]
    else:
        checkpoints = glob.glob(path + '**/*.pt', recursive=True)
    if checkpoints:
        checkpoints = sorted(checkpoints, key=natural_key)
        return checkpoints[-1]
    return None


def main(args):
    args = parse_args(args)

    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # fully initialize distributed device environment
    device = init_distributed_device(args)

    # get the name of the experiments
    if args.name is None:
        # sanitize model name for filesystem / uri use, easier if we don't use / in name as a rule?
        model_name_safe = args.model.replace('/', '-')
        date_str = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        if args.distributed:
            # sync date_str from master to all ranks
            date_str = broadcast_object(args, date_str)
        args.name = '-'.join([
            date_str,
            f"model_{model_name_safe}",
            f"lr_{args.lr}",
            f"b_{args.batch_size}",
            f"j_{args.workers}",
            f"p_{args.precision}",
        ])

    resume_latest = args.resume == 'latest'
    log_base_path = os.path.join(args.logs, args.name)
    args.log_path = None
    if is_master(args, local=args.log_local):
        os.makedirs(log_base_path, exist_ok=True)
        log_filename = f'out-{args.rank}' if args.log_local else 'out.log'
        args.log_path = os.path.join(log_base_path, log_filename)
        if os.path.exists(args.log_path) and not resume_latest:
            print(
                "Error. Experiment already exists. Use --name {} to specify a new experiment."
            )
            return -1

    # Setup text logger
    args.log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(args.log_path, args.log_level)

    if getattr(args, "samcl_train_align", False) and is_master(args):
        logging.info(
            "samcl_train_align: lr_scheduler=%s warmup=%s wd=%s lr=%s max_steps=%s (legacy src/samcl/train.py-style). "
            "Pass --lr explicitly on the command line to override lr.",
            args.lr_scheduler,
            args.warmup,
            args.wd,
            args.lr,
            getattr(args, "max_steps", None),
        )

    # Setup wandb, tensorboard, checkpoint logging
    args.wandb = 'wandb' in args.report_to or 'all' in args.report_to
    args.tensorboard = 'tensorboard' in args.report_to or 'all' in args.report_to
    args.checkpoint_path = os.path.join(log_base_path, "checkpoints")
    if is_master(args):
        args.tensorboard_path = os.path.join(log_base_path, "tensorboard") if args.tensorboard else ''
        for dirname in [args.tensorboard_path, args.checkpoint_path]:
            if dirname:
                os.makedirs(dirname, exist_ok=True)
    else:
        args.tensorboard_path = ''

    if resume_latest:
        resume_from = None
        checkpoint_path = args.checkpoint_path
        # If using remote_sync, need to check the remote instead of the local checkpoints folder.
        if args.remote_sync is not None:
            checkpoint_path = os.path.join(args.remote_sync, args.name, "checkpoints")
            if args.save_most_recent:
                print('Error. Cannot use save-most-recent with remote_sync and resume latest.')
                return -1
            if args.remote_sync_protocol != 's3':
                print('Error. Sync protocol not supported when using resume latest.')
                return -1
        if is_master(args):
            # Checking for existing checkpoint via master rank only. It is possible for
            # different rank processes to see different files if a shared file-system is under
            # stress, however it's very difficult to fully work around such situations.
            if args.save_most_recent:
                # if --save-most-recent flag is set, look for latest at a fixed filename
                resume_from = os.path.join(checkpoint_path, LATEST_CHECKPOINT_NAME)
                if not os.path.exists(resume_from):
                    # If no latest checkpoint has been saved yet, don't try to resume
                    resume_from = None
            else:
                # otherwise, list checkpoint dir contents and pick the newest checkpoint
                resume_from = get_latest_checkpoint(checkpoint_path, remote=args.remote_sync is not None)
            if resume_from:
                logging.info(f'Found latest resume checkpoint at {resume_from}.')
            else:
                logging.info(f'No latest resume checkpoint found in {checkpoint_path}.')
        if args.distributed:
            # sync found checkpoint path to all ranks
            resume_from = broadcast_object(args, resume_from)
        args.resume = resume_from

    if args.copy_codebase:
        copy_codebase(args)

    # start the sync proces if remote-sync is not None
    remote_sync_process = None
    if is_master(args) and args.remote_sync is not None:
        # first make sure it works
        result = remote_sync(
            os.path.join(args.logs, args.name), 
            os.path.join(args.remote_sync, args.name), 
            args.remote_sync_protocol
        )
        if result:
            logging.info('remote sync successful.')
        else:
            logging.info('Error: remote sync failed. Exiting.')
            return -1
        # if all looks good, start a process to do this every args.remote_sync_frequency seconds
        remote_sync_process = start_sync_process(
            args.remote_sync_frequency,
            os.path.join(args.logs, args.name), 
            os.path.join(args.remote_sync, args.name), 
            args.remote_sync_protocol
        )
        remote_sync_process.start()

    if args.precision == 'fp16':
        logging.warning(
            'It is recommended to use AMP mixed-precision instead of FP16. '
            'FP16 support needs further verification and tuning, especially for train.')

    if args.horovod:
        logging.info(
            f'Running in horovod mode with multiple processes / nodes. Device: {args.device}.'
            f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.')
    elif args.distributed:
        logging.info(
            f'Running in distributed mode with multiple processes. Device: {args.device}.'
            f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.')
    else:
        logging.info(f'Running with a single process. Device {args.device}.')

    dist_model = None
    args.distill = args.distill_model is not None and args.distill_pretrained is not None
    if args.distill:
        #FIXME: support distillation with grad accum.
        assert args.accum_freq == 1
        #FIXME: support distillation with coca.
        assert 'coca' not in args.model.lower()

    if isinstance(args.force_image_size, (tuple, list)) and len(args.force_image_size) == 1:
        # arg is nargs, single (square) image size list -> int
        args.force_image_size = args.force_image_size[0]
    random_seed(args.seed, 0)
    model_kwargs = {}
    if args.siglip:
        model_kwargs['init_logit_scale'] = np.log(10)  # different from CLIP
        model_kwargs['init_logit_bias'] = -10
    model, preprocess_train, preprocess_val = create_model_and_transforms(
        args.model,
        args.pretrained,
        precision=args.precision,
        device=device,
        jit=args.torchscript,
        force_quick_gelu=args.force_quick_gelu,
        force_custom_text=args.force_custom_text,
        force_patch_dropout=args.force_patch_dropout,
        force_image_size=args.force_image_size,
        force_context_length=args.force_context_length,
        image_mean=args.image_mean,
        image_std=args.image_std,
        image_interpolation=args.image_interpolation,
        image_resize_mode=args.image_resize_mode,  # only effective for inference
        aug_cfg=args.aug_cfg,
        pretrained_image=args.pretrained_image,
        output_dict=True,
        cache_dir=args.cache_dir,
        **model_kwargs,
    )
    if args.distill:
        # FIXME: currently assumes the model you're distilling from has the same tokenizer & transforms.
        dist_model, _, _ = create_model_and_transforms(
            args.distill_model, 
            args.distill_pretrained,
            device=device,
            precision=args.precision,
            output_dict=True,
            cache_dir=args.cache_dir,
        )
    if args.use_bnb_linear is not None:
        print('=> using a layer from bitsandbytes.\n'
              '   this is an experimental feature which requires two extra pip installs\n'
              '   pip install bitsandbytes triton'
              '   please make sure to use triton 2.0.0')
        import bitsandbytes as bnb
        from open_clip.utils import replace_linear
        print(f'=> replacing linear layers with {args.use_bnb_linear}')
        linear_replacement_cls = getattr(bnb.nn.triton_based_modules, args.use_bnb_linear)
        replace_linear(model, linear_replacement_cls)
        model = model.to(device)

    random_seed(args.seed, args.rank)

    if args.trace:
        model = trace_model(model, batch_size=args.batch_size, device=device)

    if args.lock_image:
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        model.lock_image_tower(
            unlocked_groups=args.lock_image_unlocked_groups,
            freeze_bn_stats=args.lock_image_freeze_bn_stats)
    if args.lock_text:
        model.lock_text_tower(
            unlocked_layers=args.lock_text_unlocked_layers,
            freeze_layer_norm=args.lock_text_freeze_layer_norm)

    if args.grad_checkpointing:
        model.set_grad_checkpointing()

    if is_master(args):
        logging.info("Model:")
        logging.info(f"{str(model)}")
        logging.info("Params:")
        params_file = os.path.join(args.logs, args.name, "params.txt")
        with open(params_file, "w") as f:
            for name in sorted(vars(args)):
                val = getattr(args, name)
                logging.info(f"  {name}: {val}")
                f.write(f"{name}: {val}\n")

    if args.distributed and not args.horovod:
        if args.use_bn_sync:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        ddp_args = {}
        if args.ddp_static_graph:
            # this doesn't exist in older PyTorch, arg only added if enabled
            ddp_args['static_graph'] = True
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], **ddp_args)
    
        if args.distill:
            dist_model = torch.nn.parallel.DistributedDataParallel(dist_model, device_ids=[device], **ddp_args)

    # create optimizer and scaler
    optimizer = None
    scaler = None

    if args.train_data or args.dataset_type == "synthetic":
        assert not args.trace, 'Cannot train with traced model'

        opt = getattr(args, 'opt', 'adamw').lower()
        if opt.startswith('timm/'):
            from timm.optim import create_optimizer_v2
            timm_opt = opt.split('timm/')[-1]
            opt_kwargs = {}
            assert (args.beta1 is None) == (args.beta2 is None), \
                'When using timm optimizer, BOTH beta1 and beta2 must be specified (or not specified).'
            if args.beta1 is not None:
                opt_kwargs['betas'] = (args.beta1, args.beta2)
            if args.momentum is not None:
                opt_kwargs['momentum'] = args.momentum
            optimizer = create_optimizer_v2(
                model,
                timm_opt,
                lr=args.lr,
                weight_decay=args.wd,
                eps=args.eps,
                **opt_kwargs,
            )
        else:
            # If some params are not passed, we use the default values based on model name.
            exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
            include = lambda n, p: not exclude(n, p)

            named_parameters = list(model.named_parameters())
            gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
            rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]

            if opt == 'adamw':
                optimizer = optim.AdamW(
                    [
                        {"params": gain_or_bias_params, "weight_decay": 0.},
                        {"params": rest_params, "weight_decay": args.wd},
                    ],
                    lr=args.lr,
                    betas=(args.beta1, args.beta2),
                    eps=args.eps,
                )
            else:
                assert False, f'Unknown optimizer {opt}'

        if is_master(args):
            defaults = copy.deepcopy(optimizer.defaults)
            defaults['weight_decay'] = args.wd
            defaults = ', '.join([f'{k}: {v}' for k, v in defaults.items()])
            logging.info(
                f'Created {type(optimizer).__name__} ({args.opt}) optimizer: {defaults}'
            )

        if args.horovod:
            optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
            hvd.broadcast_parameters(model.state_dict(), root_rank=0)
            hvd.broadcast_optimizer_state(optimizer, root_rank=0)

        scaler = None
        if args.precision == "amp":
            try:
                scaler = torch.amp.GradScaler(device=device)
            except (AttributeError, TypeError) as e:
                scaler = torch.cuda.amp.GradScaler()

    # optionally resume from a checkpoint
    start_epoch = 0
    resume_next_global_step = None
    resume_scheduler_total_steps = None
    resume_loaded_training = False
    if args.resume is not None:
        checkpoint = pt_load(args.resume, map_location='cpu')
        if 'epoch' in checkpoint:
            # resuming a train checkpoint w/ epoch and optimizer state
            resume_loaded_training = True
            start_epoch = checkpoint["epoch"]
            if "global_step" in checkpoint:
                resume_next_global_step = int(checkpoint["global_step"])
            if "scheduler_total_steps" in checkpoint:
                resume_scheduler_total_steps = int(checkpoint["scheduler_total_steps"])
            sd = checkpoint["state_dict"]
            if not args.distributed and next(iter(sd.items()))[0].startswith('module'):
                sd = {k[len('module.'):]: v for k, v in sd.items()}
            model.load_state_dict(sd)
            if optimizer is not None:
                optimizer.load_state_dict(checkpoint["optimizer"])
            if scaler is not None and 'scaler' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler'])
            gs_msg = f", global_step_next={resume_next_global_step}" if resume_next_global_step is not None else ""
            logging.info(f"=> resuming checkpoint '{args.resume}' (epoch {start_epoch}{gs_msg})")
        else:
            # loading a bare (model only) checkpoint for fine-tune or evaluation
            model.load_state_dict(checkpoint)
            logging.info(f"=> loaded checkpoint '{args.resume}' (epoch {start_epoch})")

    # initialize datasets
    tokenizer = get_tokenizer(args.model, cache_dir=args.cache_dir, context_length=args.force_context_length)
    if getattr(args, "coco_random_baseline_dataloader", False):
        from open_clip_train.data import get_dataset_fn, get_imagenet
        from open_clip_train.data_semantic import coco_semantic_image_root, get_coco_random_baseline_train_data

        if getattr(args, "semantic_sampling", False):
            logging.error("--coco-random-baseline-dataloader cannot be used together with --semantic-sampling.")
            return -1
        if not getattr(args, "coco_captions_json", None):
            logging.error("--coco-random-baseline-dataloader requires --coco-captions-json.")
            return -1
        try:
            img_root = coco_semantic_image_root(args)
        except ValueError as e:
            logging.error("%s", e)
            return -1
        if not os.path.isdir(img_root):
            logging.error("--coco-random-baseline-dataloader: COCO image root is not a directory: %s", img_root)
            return -1
        cap_json = str(getattr(args, "coco_captions_json", "") or "")
        if not os.path.isfile(os.path.abspath(cap_json)):
            logging.error(
                "--coco-random-baseline-dataloader: --coco-captions-json must be an existing file: %s",
                cap_json,
            )
            return -1

        data = {}
        data["train"] = get_coco_random_baseline_train_data(args, preprocess_train, tokenizer)
        if args.val_data:
            data["val"] = get_dataset_fn(args.val_data, "csv")(
                args, preprocess_val, is_train=False, tokenizer=tokenizer
            )
        if args.imagenet_val is not None:
            data["imagenet-val"] = get_imagenet(args, (preprocess_train, preprocess_val), "val")
        if args.imagenet_v2 is not None:
            data["imagenet-v2"] = get_imagenet(args, (preprocess_train, preprocess_val), "v2")
    elif getattr(args, "semantic_sampling", False):
        from open_clip_train.data_semantic import coco_semantic_image_root, semantic_uses_coco_dataset
        from open_clip_train.samcl_ext.pairs_dataset import semantic_train_data_is_wds_dir

        if semantic_uses_coco_dataset(args):
            try:
                img_root = coco_semantic_image_root(args)
            except ValueError as e:
                logging.error("%s", e)
                return -1
            if not os.path.isdir(img_root):
                logging.error("--semantic-sampling COCO: image root is not a directory: %s", img_root)
                return -1
            cap_json = str(getattr(args, "coco_captions_json", "") or "")
            if not os.path.isfile(os.path.abspath(cap_json)):
                logging.error("--semantic-sampling COCO: --coco-captions-json must be an existing file: %s", cap_json)
                return -1
        elif not args.train_data:
            logging.error("--semantic-sampling requires --train-data (CSV/TSV file or WebDataset shards directory).")
            return -1
        else:
            _is_wds_root = semantic_train_data_is_wds_dir(args.train_data)
            if not _is_wds_root:
                ext = str(args.train_data).split(".")[-1].lower()
                if ext not in ("csv", "tsv", "txt"):
                    logging.error(
                        "--semantic-sampling: --train-data must be a directory of .tar shards "
                        "(e.g. cc3m_wds/) or a .csv/.tsv/.txt table with image paths and captions; "
                        "or use --coco-captions-json (+ image root) for COCO."
                    )
                    return -1
        from open_clip_train.data import get_dataset_fn, get_imagenet
        from open_clip_train.data_semantic import get_semantic_train_data

        data = {}
        data["train"] = get_semantic_train_data(args, preprocess_train, tokenizer)
        if args.val_data:
            data["val"] = get_dataset_fn(args.val_data, "csv")(
                args, preprocess_val, is_train=False, tokenizer=tokenizer
            )
        if args.imagenet_val is not None:
            data["imagenet-val"] = get_imagenet(args, (preprocess_train, preprocess_val), "val")
        if args.imagenet_v2 is not None:
            data["imagenet-v2"] = get_imagenet(args, (preprocess_train, preprocess_val), "v2")
    else:
        data = get_data(
            args,
            (preprocess_train, preprocess_val),
            epoch=start_epoch,
            tokenizer=tokenizer,
        )
    assert len(data), 'At least one train or eval dataset must be specified.'

    # LR scheduler step offset (updated each epoch in train_one_epoch). Mid-epoch checkpoints store global_step.
    if "train" in data:
        _nb_pe = data["train"].dataloader.num_batches // args.accum_freq
        if resume_next_global_step is not None:
            args._scheduler_step_offset = int(resume_next_global_step)
        elif resume_loaded_training:
            args._scheduler_step_offset = int(start_epoch) * int(_nb_pe)
        else:
            args._scheduler_step_offset = 0

    # create scheduler if train
    scheduler = None
    if 'train' in data and optimizer is not None:
        total_steps, nb_lr = _compute_scheduler_total_steps(
            args, data["train"].dataloader, resume_scheduler_total_steps
        )
        if getattr(args, "max_steps", None):
            total_steps = max(int(total_steps), int(args.max_steps))
        args.scheduler_total_steps = int(total_steps)
        if resume_next_global_step is not None and int(resume_next_global_step) >= int(total_steps):
            logging.warning(
                "global_step (%s) >= scheduler_total_steps (%s): learning rate will stay at cosine minimum (0) "
                "until you increase --epochs/--train-num-samples or resume with a checkpoint that stores a larger "
                "scheduler_total_steps.",
                resume_next_global_step,
                total_steps,
            )
        if args.lr_scheduler == "cosine":
            scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)
        elif args.lr_scheduler == "const":
            scheduler = const_lr(optimizer, args.lr, args.warmup, total_steps)
        elif args.lr_scheduler == "const-cooldown":
            assert args.epochs_cooldown is not None,\
                "Please specify the number of cooldown epochs for this lr schedule."
            acc = max(1, int(args.accum_freq))
            cooldown_steps = (nb_lr // acc) * args.epochs_cooldown
            scheduler = const_lr_cooldown(
                optimizer, args.lr, args.warmup, total_steps,
                cooldown_steps, args.lr_cooldown_power, args.lr_cooldown_end)
        else:
            logging.error(
                f'Unknown scheduler, {args.lr_scheduler}. Available options are: cosine, const, const-cooldown.')
            exit(1)
        if resume_scheduler_total_steps is None:
            logging.info(
                "LR schedule: scheduler_total_steps=%d (epochs=%d, warmup=%d, accum_freq=%d).",
                int(total_steps),
                int(args.epochs),
                int(args.warmup),
                int(args.accum_freq),
            )

    # determine if this worker should save logs and checkpoints. only do so if it is rank == 0
    args.save_logs = args.logs and args.logs.lower() != 'none' and is_master(args)
    writer = None
    if args.save_logs and args.tensorboard:
        assert tensorboard is not None, "Please install tensorboard."
        writer = tensorboard.SummaryWriter(args.tensorboard_path)

    if args.wandb and is_master(args):
        assert wandb is not None, 'Please install wandb.'
        logging.debug('Starting wandb.')
        args.train_sz = data["train"].dataloader.num_samples
        if args.val_data is not None:
            args.val_sz = data["val"].dataloader.num_samples
        # you will have to configure this for your project!
        wandb.init(
            project=args.wandb_project_name,
            name=args.name,
            id=args.name,
            notes=args.wandb_notes,
            tags=[],
            resume='auto' if args.resume == "latest" else None,
            config=vars(args),
        )
        if args.debug:
            wandb.watch(model, log='all')
        wandb.save(params_file)
        logging.debug('Finished loading wandb.')

    # Pytorch 2.0 adds '_orig_mod.' prefix to keys of state_dict() of compiled models.
    # For compatibility, we save state_dict() of the original model, which shares the
    # weights without the prefix.
    original_model = model
    if args.torchcompile:
        logging.info('Compiling model...')

        if args.grad_checkpointing and args.distributed:
            logging.info('Disabling DDP dynamo optimizer when grad checkpointing enabled.')
            # As of now (~PyTorch 2.4/2.5), compile + grad checkpointing work, but DDP optimizer must be disabled
            torch._dynamo.config.optimize_ddp = False

        filter_prefixes = (
            "torch._dynamo",
            "torch._inductor",
            "torch._functorch",
            "torch._utils_internal",
            "torch.fx",
        )

        for name in logging.root.manager.loggerDict:
            if name.startswith(filter_prefixes):
                logging.getLogger(name).setLevel(logging.WARNING)

        model = torch.compile(original_model)

    if 'train' not in data:
        # If using int8, convert to inference mode.
        if args.use_bnb_linear is not None:
            from open_clip.utils import convert_int8_model_to_inference_mode
            convert_int8_model_to_inference_mode(model)
        # Evaluate.
        evaluate(model, data, start_epoch, args, tb_writer=writer, tokenizer=tokenizer)
        return

    loss = create_loss(args)

    for epoch in range(start_epoch, args.epochs):
        if is_master(args):
            logging.info(f'Start epoch {epoch}')

        train_one_epoch(
            model,
            data,
            loss,
            epoch,
            optimizer,
            scaler,
            scheduler,
            dist_model,
            args,
            tb_writer=writer,
            preprocess_eval=preprocess_val,
            tokenizer=tokenizer,
        )
        completed_epoch = epoch + 1

        stop_max = getattr(args, "_stop_training_after_max_steps", False)
        if stop_max and is_master(args):
            logging.info(
                "Stopping training: --max-steps=%s reached (next global_step=%s).",
                getattr(args, "max_steps", None),
                getattr(args, "_scheduler_step_offset", None),
            )

        if (not stop_max) and any(v in data for v in ('val', 'imagenet-val', 'imagenet-v2')):
            evaluate(model, data, completed_epoch, args, tb_writer=writer, tokenizer=tokenizer)
            # sync to avoid some processes advancing/exiting while rank 0 finishes eval
            if args.distributed:
                if args.horovod:
                    hvd.join()
                else:
                    torch.distributed.barrier()

        # Saving checkpoints.
        if args.save_logs:
            checkpoint_dict = {
                "epoch": completed_epoch,
                "name": args.name,
                "state_dict": original_model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            if getattr(args, "scheduler_total_steps", None) is not None:
                checkpoint_dict["scheduler_total_steps"] = int(args.scheduler_total_steps)
            if scaler is not None:
                checkpoint_dict["scaler"] = scaler.state_dict()

            if completed_epoch == args.epochs or (
                args.save_frequency > 0 and (completed_epoch % args.save_frequency) == 0
            ):
                torch.save(
                    checkpoint_dict,
                    os.path.join(args.checkpoint_path, f"epoch_{completed_epoch}.pt"),
                )
            if args.delete_previous_checkpoint:
                previous_epoch = completed_epoch - args.save_frequency
                if previous_epoch > 0:
                    previous_checkpoint = os.path.join(args.checkpoint_path, f"epoch_{previous_epoch}.pt")
                    if os.path.exists(previous_checkpoint):
                        os.remove(previous_checkpoint)

            if args.save_most_recent:
                checkpoint_dict["global_step"] = int(getattr(args, "_scheduler_step_offset", 0) or 0)
                if getattr(args, "scheduler_total_steps", None) is not None:
                    checkpoint_dict["scheduler_total_steps"] = int(args.scheduler_total_steps)
                # try not to corrupt the latest checkpoint if save fails
                tmp_save_path = os.path.join(args.checkpoint_path, "tmp.pt")
                latest_save_path = os.path.join(args.checkpoint_path, LATEST_CHECKPOINT_NAME)
                torch.save(checkpoint_dict, tmp_save_path)
                os.replace(tmp_save_path, latest_save_path)

        # keep nodes in sync during checkpointing
        if args.distributed:
            if args.horovod:
                hvd.join()
            else:
                torch.distributed.barrier()

        if getattr(args, "_stop_training_after_max_steps", False):
            break

    if args.wandb and is_master(args):
        wandb.finish()

    # run a final sync.
    if remote_sync_process is not None:
        logging.info('Final remote sync.')
        remote_sync_process.terminate()
        result = remote_sync(
            os.path.join(args.logs, args.name), 
            os.path.join(args.remote_sync, args.name), 
            args.remote_sync_protocol
        )
        if result:
            logging.info('Final remote sync successful.')
        else:
            logging.info('Final remote sync failed.')
    

def copy_codebase(args):
    from shutil import copytree, ignore_patterns
    new_code_path = os.path.join(args.logs, args.name, "code")
    if os.path.exists(new_code_path):
        print(
            f"Error. Experiment already exists at {new_code_path}. Use --name to specify a new experiment."
        )
        return -1
    print(f"Copying codebase to {new_code_path}")
    current_code_path = os.path.realpath(__file__)
    for _ in range(3):
        current_code_path = os.path.dirname(current_code_path)
    copytree(current_code_path, new_code_path, ignore=ignore_patterns('log', 'logs', 'wandb'))
    print("Done copying code.")
    return 1


if __name__ == "__main__":
    main(sys.argv[1:])
