from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Iterator, Union

import torch
from torch.utils.data import BatchSampler

from open_clip_train.samcl_ext.coco_pairs import CocoPairsDataset
from open_clip_train.samcl_ext.pairs_dataset import CsvPairsDataset, WdsPairsDataset
from open_clip_train.samcl_ext.relations import SemanticRelationOracle


@dataclass(frozen=True)
class SemanticMix:
    r1: float = 0.25
    r2: float = 0.25
    r3: float = 0.25
    r4: float = 0.25

    def normalized(self) -> "SemanticMix":
        s = float(self.r1 + self.r2 + self.r3 + self.r4)
        if s <= 0:
            return SemanticMix(0.25, 0.25, 0.25, 0.25)
        return SemanticMix(self.r1 / s, self.r2 / s, self.r3 / s, self.r4 / s)

    def sample_relation(self, rng: random.Random) -> int:
        m = self.normalized()
        x = rng.random()
        if x < m.r1:
            return 1
        if x < m.r1 + m.r2:
            return 2
        if x < m.r1 + m.r2 + m.r3:
            return 3
        return 4

    def probs(self) -> dict[int, float]:
        m = self.normalized()
        return {1: float(m.r1), 2: float(m.r2), 3: float(m.r3), 4: float(m.r4)}


@dataclass(frozen=True)
class BinaryMix:
    """
    2-way relation mix for text_only / image_only experiments:
      - 1: similar (text or image)
      - 2: different (text or image)
    """

    similar: float = 0.5
    different: float = 0.5

    def normalized(self) -> "BinaryMix":
        s = float(self.similar + self.different)
        if s <= 0:
            return BinaryMix(0.5, 0.5)
        return BinaryMix(self.similar / s, self.different / s)

    def sample_relation(self, rng: random.Random) -> int:
        m = self.normalized()
        return 1 if rng.random() < float(m.similar) else 2

    def probs(self) -> dict[int, float]:
        m = self.normalized()
        return {1: float(m.similar), 2: float(m.different)}


class RandomBatchSampler(BatchSampler):
    """
    Baseline: uniform random batches over dataset indices.
    """

    def __init__(
        self,
        dataset: Union[CsvPairsDataset, WdsPairsDataset, CocoPairsDataset],
        *,
        batch_size: int,
        drop_last: bool = True,
        seed: int = 0,
    ):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[list[Any]]:
        rng = random.Random(self.seed + self.epoch)
        indices = list(range(len(self.dataset)))
        rng.shuffle(indices)

        batch: list[Any] = []
        for idx in indices:
            # Attach meta so training can log sampler behavior consistently.
            batch.append(
                (
                    int(idx),
                    {"is_fallback": 1, "tries": 0, "target_relation": -1, "found_relation": -1},
                )
            )
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch and (not self.drop_last):
            yield batch

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


class SemanticBatchSampler(BatchSampler):
    """
    Semantic-aware batch composition.

    Important: we do NOT change the dataset pairs. We only choose which positive pairs
    co-occur in the same batch, so that cross-pairs (i_a, t_b) exhibit desired relation types.
    """

    def __init__(
        self,
        dataset: Union[CsvPairsDataset, WdsPairsDataset, CocoPairsDataset],
        *,
        oracle: SemanticRelationOracle,
        mix: SemanticMix | BinaryMix,
        relation_mode: str = "full",
        mode: str = "single",
        num_anchors: int = 4,
        min_anchor_matches: int | None = None,
        global_num_candidates: int = 64,
        num_blocks: int = 1,
        compute_device: str = "cpu",
        cache_teacher_on_device: bool = True,
        batch_size: int,
        drop_last: bool = True,
        seed: int = 0,
        max_tries: int = 1000,
    ) -> None:
        self.dataset = dataset
        self.oracle = oracle
        self.relation_mode = str(relation_mode).lower().strip()
        self.mix = mix.normalized()
        self.mode = str(mode)
        self.num_anchors = int(num_anchors)
        self.min_anchor_matches = int(min_anchor_matches) if min_anchor_matches is not None else None
        self.global_num_candidates = int(global_num_candidates)
        self.num_blocks = int(num_blocks)
        # Normalize compute device. torch.device("cuda") (index=None) does NOT compare equal to cuda:0,
        # so we resolve it to a concrete device index.
        dev = torch.device(str(compute_device))
        if dev.type == "cuda":
            if not torch.cuda.is_available():
                dev = torch.device("cpu")
            elif dev.index is None:
                dev = torch.device(f"cuda:{torch.cuda.current_device()}")
        self.compute_device = dev
        self.cache_teacher_on_device = bool(cache_teacher_on_device)
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0
        self.max_tries = int(max_tries)

        # Fast lookup: caption_id -> dataset index; image_id -> list[dataset indices]
        self.caption_id_to_index: dict[int, int] = {}
        self.image_id_to_indices: dict[int, list[int]] = {}
        for idx, p in enumerate(dataset.pairs):
            self.caption_id_to_index[int(p.caption_id)] = int(idx)
            self.image_id_to_indices.setdefault(int(p.image_id), []).append(int(idx))
        for k in list(self.image_id_to_indices.keys()):
            self.image_id_to_indices[k].sort()

        self._all_indices = list(range(len(dataset)))
        self._all_image_ids = dataset.image_ids

        # Small caches to speed up global/multi modes
        self._image_id_to_caption_rows: dict[int, torch.Tensor] = {}
        self._image_id_to_caption_rows_dev: dict[int, torch.Tensor] = {}

        self._caption_emb_dev: torch.Tensor | None = None
        self._image_emb_dev: torch.Tensor | None = None

    @dataclass
    class _GlobalState:
        image_ids: list[int]
        caption_ids: list[int]
        img_emb: torch.Tensor  # [k, Di] float32
        cap_emb: torch.Tensor  # [k, Dt] float32
        cap_img_emb: torch.Tensor  # [k, Di] float32
        capset_rows: torch.Tensor  # [k, cmax] long (caption embedding rows per image)
        capset_mask: torch.Tensor  # [k, cmax] bool
        capset_emb: torch.Tensor  # [k, cmax, Dt] float32
        cmax: int

    def _teacher_caption_emb(self) -> torch.Tensor:
        tc = self.oracle.teacher_cache
        assert tc.caption_emb is not None
        if self.compute_device.type == "cpu":
            return tc.caption_emb
        if (self._caption_emb_dev is None) or (self._caption_emb_dev.device != self.compute_device):
            if not self.cache_teacher_on_device:
                # We'll still move slices on demand elsewhere; keep full cache on CPU.
                return tc.caption_emb
            self._caption_emb_dev = tc.caption_emb.to(self.compute_device, non_blocking=True)
        return self._caption_emb_dev

    def _teacher_image_emb(self) -> torch.Tensor:
        tc = self.oracle.teacher_cache
        assert tc.image_emb is not None
        if self.compute_device.type == "cpu":
            return tc.image_emb
        if (self._image_emb_dev is None) or (self._image_emb_dev.device != self.compute_device):
            if not self.cache_teacher_on_device:
                return tc.image_emb
            self._image_emb_dev = tc.image_emb.to(self.compute_device, non_blocking=True)
        return self._image_emb_dev

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def _random_index(self, rng: random.Random) -> int:
        return int(self._all_indices[rng.randrange(0, len(self._all_indices))])

    def _random_image_id(self, rng: random.Random) -> int:
        return int(self._all_image_ids[rng.randrange(0, len(self._all_image_ids))])

    def _random_index_from_image(self, image_id: int, rng: random.Random) -> int:
        lst = self.image_id_to_indices[int(image_id)]
        return int(lst[rng.randrange(0, len(lst))])

    def _caption_rows_for_image(self, image_id: int) -> torch.Tensor:
        """
        Returns caption embedding row indices for all GT captions of this image_id.
        Cached for efficiency (COCO typically has 5 captions per image).
        """
        image_id = int(image_id)
        if image_id in self._image_id_to_caption_rows:
            return self._image_id_to_caption_rows[image_id]
        cap_ids = self.dataset.caption_ids_for_image(image_id)
        tc = self.oracle.teacher_cache
        assert tc.caption_id_to_row is not None
        rows = torch.tensor([int(tc.caption_id_to_row[int(cid)]) for cid in cap_ids], dtype=torch.long)
        self._image_id_to_caption_rows[image_id] = rows
        return rows

    def _caption_rows_for_image_dev(self, image_id: int) -> torch.Tensor:
        image_id = int(image_id)
        if self.compute_device.type == "cpu":
            return self._caption_rows_for_image(image_id)
        if image_id in self._image_id_to_caption_rows_dev:
            return self._image_id_to_caption_rows_dev[image_id]
        rows = self._caption_rows_for_image(image_id).to(self.compute_device, non_blocking=True)
        self._image_id_to_caption_rows_dev[image_id] = rows
        return rows

    def _image_emb(self, image_ids: list[int]) -> torch.Tensor:
        tc = self.oracle.teacher_cache
        assert tc.image_id_to_row is not None
        rows = torch.tensor([int(tc.image_id_to_row[int(i)]) for i in image_ids], dtype=torch.long, device=self.compute_device)
        emb = self._teacher_image_emb()
        if emb.device != self.compute_device:
            # full cache is on CPU, but we want compute on device; move slice.
            return emb.index_select(0, rows.cpu()).to(self.compute_device).float().contiguous()
        return emb.index_select(0, rows).float().contiguous()

    def _caption_emb(self, caption_ids: list[int]) -> torch.Tensor:
        tc = self.oracle.teacher_cache
        assert tc.caption_id_to_row is not None
        rows = torch.tensor([int(tc.caption_id_to_row[int(c)]) for c in caption_ids], dtype=torch.long, device=self.compute_device)
        emb = self._teacher_caption_emb()
        if emb.device != self.compute_device:
            return emb.index_select(0, rows.cpu()).to(self.compute_device).float().contiguous()
        return emb.index_select(0, rows).float().contiguous()

    def _build_global_state(self, image_ids: list[int], caption_ids: list[int]) -> "_GlobalState":
        """
        Build cached tensors for global-mode scoring for the current partial batch.
        This is called repeatedly as the batch grows, so we keep it lightweight and
        avoid recomputing per-candidate work.
        """
        k = int(len(image_ids))
        assert k == int(len(caption_ids))
        tc = self.oracle.teacher_cache
        assert tc.caption_emb is not None

        img_emb = self._image_emb(image_ids)  # [k, Di]
        cap_emb = self._caption_emb(caption_ids)  # [k, Dt]

        # In this dataset, each caption belongs to its paired image.
        cap_img_emb = self._image_emb([int(self.dataset.image_id_for_caption(int(cid))) for cid in caption_ids])

        rows_list = [self._caption_rows_for_image_dev(int(iid)) for iid in image_ids]
        cmax = max((int(r.numel()) for r in rows_list), default=0)
        if cmax <= 0:
            capset_rows = torch.empty((k, 0), dtype=torch.long, device=self.compute_device)
            capset_mask = torch.empty((k, 0), dtype=torch.bool, device=self.compute_device)
            capset_emb = torch.empty((k, 0, cap_emb.shape[-1]), dtype=torch.float32)
            return self._GlobalState(
                image_ids=list(image_ids),
                caption_ids=list(caption_ids),
                img_emb=img_emb,
                cap_emb=cap_emb,
                cap_img_emb=cap_img_emb,
                capset_rows=capset_rows,
                capset_mask=capset_mask,
                capset_emb=capset_emb,
                cmax=0,
            )

        capset_rows = torch.full((k, cmax), -1, dtype=torch.long, device=self.compute_device)
        capset_mask = torch.zeros((k, cmax), dtype=torch.bool, device=self.compute_device)
        for ii, rr in enumerate(rows_list):
            n = int(rr.numel())
            if n > 0:
                capset_rows[ii, :n] = rr
                capset_mask[ii, :n] = True

        flat = capset_rows.clamp_min(0).reshape(-1)
        cap_emb_all = self._teacher_caption_emb()
        if cap_emb_all.device != self.compute_device:
            capset_emb = cap_emb_all.index_select(0, flat.cpu()).to(self.compute_device).float().view(k, cmax, -1)
        else:
            capset_emb = cap_emb_all.index_select(0, flat).float().view(k, cmax, -1)  # [k, cmax, Dt]
        return self._GlobalState(
            image_ids=list(image_ids),
            caption_ids=list(caption_ids),
            img_emb=img_emb,
            cap_emb=cap_emb,
            cap_img_emb=cap_img_emb,
            capset_rows=capset_rows,
            capset_mask=capset_mask,
            capset_emb=capset_emb,
            cmax=int(cmax),
        )

    def _global_delta_from_state(self, *, cand_idx: int, state: "_GlobalState") -> tuple[dict[int, int], int]:
        """
        Vectorized delta computation using cached state. Threshold-based image similarity only.
        """
        cand = self.dataset.pairs[int(cand_idx)]
        i_new = int(cand.image_id)
        t_new = int(cand.caption_id)

        k = int(len(state.image_ids))
        text_thr = float(self.oracle.cfg.text_sim_threshold)
        img_thr = float(self.oracle.cfg.image_sim_threshold)

        # Candidate embeddings
        e_t_new = self._caption_emb([int(t_new)]).view(-1)  # [Dt]
        e_i_new = self._image_emb([int(i_new)]).view(-1)  # [Di]

        # -------- (i_existing, t_new) side --------
        sim_img_1 = (state.img_emb @ e_i_new) >= img_thr  # [k]
        if int(state.cmax) <= 0:
            sim_txt_1 = torch.zeros((k,), dtype=torch.bool)
        else:
            sims = torch.matmul(state.capset_emb, e_t_new)  # [k, cmax]
            sims = sims.masked_fill(~state.capset_mask, -1e9)
            sim_txt_1 = sims.max(dim=1).values >= text_thr  # [k]

        if self.relation_mode == "full":
            r1 = int((sim_txt_1 & sim_img_1).sum().item())
            r2 = int((sim_txt_1 & (~sim_img_1)).sum().item())
            r3 = int(((~sim_txt_1) & sim_img_1).sum().item())
            r4 = int(((~sim_txt_1) & (~sim_img_1)).sum().item())
            counts_side1 = {1: r1, 2: r2, 3: r3, 4: r4}
        elif self.relation_mode == "text_only":
            s = int(sim_txt_1.sum().item())
            counts_side1 = {1: s, 2: int(k - s)}
        elif self.relation_mode == "image_only":
            s = int(sim_img_1.sum().item())
            counts_side1 = {1: s, 2: int(k - s)}
        else:
            raise ValueError(f"Unknown relation_mode: {self.relation_mode!r}")

        # -------- (i_new, t_existing) side --------
        sim_img_2 = (state.cap_img_emb @ e_i_new) >= img_thr  # [k]
        rows_new = self._caption_rows_for_image_dev(int(i_new))
        if int(rows_new.numel()) <= 0:
            sim_txt_2 = torch.zeros((k,), dtype=torch.bool)
        else:
            cap_emb_all = self._teacher_caption_emb()
            if cap_emb_all.device != self.compute_device:
                capset = cap_emb_all.index_select(0, rows_new.cpu()).to(self.compute_device).float()
            else:
                capset = cap_emb_all.index_select(0, rows_new).float()  # [c, Dt]
            sims2 = torch.matmul(state.cap_emb, capset.T)  # [k, c]
            sim_txt_2 = sims2.max(dim=1).values >= text_thr  # [k]

        if self.relation_mode == "full":
            r1b = int((sim_txt_2 & sim_img_2).sum().item())
            r2b = int((sim_txt_2 & (~sim_img_2)).sum().item())
            r3b = int(((~sim_txt_2) & sim_img_2).sum().item())
            r4b = int(((~sim_txt_2) & (~sim_img_2)).sum().item())
            counts_side2 = {1: r1b, 2: r2b, 3: r3b, 4: r4b}
        elif self.relation_mode == "text_only":
            s = int(sim_txt_2.sum().item())
            counts_side2 = {1: s, 2: int(k - s)}
        elif self.relation_mode == "image_only":
            s = int(sim_img_2.sum().item())
            counts_side2 = {1: s, 2: int(k - s)}
        else:
            raise ValueError(f"Unknown relation_mode: {self.relation_mode!r}")

        rel_keys = sorted(self.mix.probs().keys())
        counts = {r: int(counts_side1.get(r, 0) + counts_side2.get(r, 0)) for r in rel_keys}
        delta_total = int(2 * k)
        return counts, delta_total

    def _propose_index(self, anchor_image_id: int, relation: int, *, rng: random.Random) -> int:
        """
        Propose a candidate dataset index, using lightweight heuristics:
          - For relation 1/3 (needs similar image), prefer sampling from similar-image pool if enabled.
          - For relation 2/4, uniform random over dataset.
        """
        need_sim_image = False
        if self.relation_mode == "full":
            need_sim_image = relation in (1, 3)
        elif self.relation_mode == "image_only":
            need_sim_image = relation == 1
        # text_only has no image constraint; treat as random proposals.

        if need_sim_image:
            if self.oracle.cfg.use_image_topk and getattr(self.oracle, "_img_topk", None) is not None:
                nn = self.oracle._img_topk.neighbors(int(anchor_image_id))  # noqa: SLF001 (research code)
                cand_image_id = int(nn[rng.randrange(0, len(nn))].item())
            else:
                cand_image_id = self._random_image_id(rng)
            return self._random_index_from_image(cand_image_id, rng)
        return self._random_index(rng)

    def _find_candidate_single(
        self, anchor_image_id: int, relation: int, *, rng: random.Random, used: set[int]
    ) -> tuple[int, dict[str, int]]:
        anchor_image_id = int(anchor_image_id)

        tries = 0
        for _ in range(self.max_tries):
            tries += 1
            cand_idx = self._propose_index(anchor_image_id, relation, rng=rng)

            if cand_idx in used:
                continue

            cand = self.dataset.pairs[cand_idx]
            r = self.oracle.get_relation(anchor_image_id, int(cand.caption_id), mode=self.relation_mode)
            if r == relation:
                return int(cand_idx), {
                    "is_fallback": 0,
                    "tries": int(tries),
                    "target_relation": int(relation),
                    "found_relation": int(r),
                }

        # Fallback: any unused random index
        for _ in range(self.max_tries):
            cand_idx = self._random_index(rng)
            if cand_idx not in used:
                # Record the actually realized relation for diagnostics.
                cand = self.dataset.pairs[cand_idx]
                r = self.oracle.get_relation(anchor_image_id, int(cand.caption_id), mode=self.relation_mode)
                return int(cand_idx), {
                    "is_fallback": 1,
                    "tries": int(tries),
                    "target_relation": int(relation),
                    "found_relation": int(r),
                }

        # Worst-case: allow reuse (should be rare). Still emit meta.
        cand_idx = self._random_index(rng)
        cand = self.dataset.pairs[cand_idx]
        r = self.oracle.get_relation(anchor_image_id, int(cand.caption_id), mode=self.relation_mode)
        return int(cand_idx), {
            "is_fallback": 1,
            "tries": int(tries),
            "target_relation": int(relation),
            "found_relation": int(r),
        }

    def _find_candidate_multi(
        self,
        anchor_image_ids: list[int],
        relation: int,
        *,
        rng: random.Random,
        used: set[int],
    ) -> tuple[int, dict[str, int]]:
        assert len(anchor_image_ids) > 0
        anchor_image_ids = [int(x) for x in anchor_image_ids]
        relation = int(relation)
        tries = 0
        min_matches = self.min_anchor_matches
        if min_matches is None:
            # default: require a majority of anchors to match
            min_matches = max(1, (len(anchor_image_ids) + 1) // 2)

        # For higher success rate, don't always guide proposals from the first anchor.
        # Instead, on each try pick a random guide anchor. When use_image_topk is enabled,
        # we additionally sample from the union of all anchors' neighbor pools for relations (1,3).
        guide_anchor = int(anchor_image_ids[0])

        nn_pool: torch.Tensor | None = None
        need_sim_image = False
        if self.relation_mode == "full":
            need_sim_image = relation in (1, 3)
        elif self.relation_mode == "image_only":
            need_sim_image = relation == 1

        if need_sim_image and bool(self.oracle.cfg.use_image_topk) and getattr(self.oracle, "_img_topk", None) is not None:
            try:
                nns = [self.oracle._img_topk.neighbors(int(aid)) for aid in anchor_image_ids]  # noqa: SLF001
                if nns:
                    nn_pool = torch.cat(nns, dim=0)
            except Exception:
                nn_pool = None

        for _ in range(self.max_tries):
            tries += 1
            guide_anchor = int(anchor_image_ids[rng.randrange(0, len(anchor_image_ids))])

            if nn_pool is not None and int(nn_pool.numel()) > 0:
                cand_image_id = int(nn_pool[rng.randrange(0, int(nn_pool.numel()))].item())
                cand_idx = self._random_index_from_image(cand_image_id, rng)
            else:
                cand_idx = self._propose_index(guide_anchor, relation, rng=rng)

            if cand_idx in used:
                continue
            cand = self.dataset.pairs[cand_idx]
            cid = int(cand.caption_id)
            matches = 0
            for aid in anchor_image_ids:
                if self.oracle.get_relation(int(aid), cid, mode=self.relation_mode) == relation:
                    matches += 1
                    if matches >= int(min_matches):
                        return int(cand_idx), {
                            "is_fallback": 0,
                            "tries": int(tries),
                            "target_relation": int(relation),
                            "found_relation": int(relation),
                        }

        # Fallback: any unused random index
        for _ in range(self.max_tries):
            cand_idx = self._random_index(rng)
            if cand_idx not in used:
                cand = self.dataset.pairs[cand_idx]
                r = self.oracle.get_relation(guide_anchor, int(cand.caption_id), mode=self.relation_mode)
                return int(cand_idx), {
                    "is_fallback": 1,
                    "tries": int(tries),
                    "target_relation": int(relation),
                    "found_relation": int(r),
                }

        cand_idx = self._random_index(rng)
        cand = self.dataset.pairs[cand_idx]
        r = self.oracle.get_relation(guide_anchor, int(cand.caption_id), mode=self.relation_mode)
        return int(cand_idx), {
            "is_fallback": 1,
            "tries": int(tries),
            "target_relation": int(relation),
            "found_relation": int(r),
        }

    def _global_score(
        self,
        *,
        cur_counts: dict[int, int],
        cur_total: int,
        delta_counts: dict[int, int],
        delta_total: int,
    ) -> float:
        """
        L1 distance between new distribution and target mix.
        """
        tgt = self.mix.probs()
        denom = float(max(1, int(cur_total + delta_total)))
        score = 0.0
        for r in tgt.keys():
            p = float(cur_counts.get(r, 0) + delta_counts.get(r, 0)) / denom
            score += abs(p - tgt[r])
        return float(score)

    def _global_delta(
        self,
        *,
        cand_idx: int,
        cur_image_ids: list[int],
        cur_caption_ids: list[int],
    ) -> tuple[dict[int, int], int]:
        """
        If we add candidate pair (i_new, t_new), we create new cross edges:
          - (i_new, t_existing) for all existing captions
          - (i_existing, t_new) for all existing images
        We count ordered pairs, matching the training histogram definition.
        """
        # If image similarity uses top-k neighbor membership, fall back to the oracle (slower).
        if bool(self.oracle.cfg.use_image_topk):
            cand = self.dataset.pairs[int(cand_idx)]
            i_new = int(cand.image_id)
            t_new = int(cand.caption_id)
            rel_keys = sorted(self.mix.probs().keys())
            counts = {int(k): 0 for k in rel_keys}
            for i in cur_image_ids:
                r = self.oracle.get_relation(int(i), int(t_new), mode=self.relation_mode)
                counts[int(r)] += 1
            for t in cur_caption_ids:
                r = self.oracle.get_relation(int(i_new), int(t), mode=self.relation_mode)
                counts[int(r)] += 1
            delta_total = int(len(cur_image_ids) + len(cur_caption_ids))
            return counts, delta_total

        # Fast vectorized path (threshold-based image similarity).
        state = self._build_global_state([int(x) for x in cur_image_ids], [int(x) for x in cur_caption_ids])
        return self._global_delta_from_state(cand_idx=int(cand_idx), state=state)

    def _find_candidate_global(
        self,
        *,
        rng: random.Random,
        used: set[int],
        cur_counts: dict[int, int],
        cur_total: int,
        cur_image_ids: list[int],
        cur_caption_ids: list[int],
    ) -> tuple[int, dict[str, int], dict[int, int], int]:
        """
        Greedy batch-level selection:
          sample a proposal set and pick the candidate that best matches target mix
          on the *full* batch cross-pair histogram (L1 distance).
        """
        # If image similarity uses top-k neighbor membership, fall back to the oracle (slower but exact).
        if bool(self.oracle.cfg.use_image_topk):
            best_idx: int | None = None
            best_score: float | None = None
            best_delta: dict[int, int] | None = None
            best_delta_total: int = 0
            evaluated = 0
            for _ in range(max(1, self.global_num_candidates)):
                cand_idx = self._random_index(rng)
                if cand_idx in used:
                    continue
                evaluated += 1
                delta, delta_total = self._global_delta(
                    cand_idx=int(cand_idx), cur_image_ids=cur_image_ids, cur_caption_ids=cur_caption_ids
                )
                score = self._global_score(
                    cur_counts=cur_counts,
                    cur_total=cur_total,
                    delta_counts=delta,
                    delta_total=delta_total,
                )
                if best_score is None or score < best_score:
                    best_score = float(score)
                    best_idx = int(cand_idx)
                    best_delta = delta
                    best_delta_total = int(delta_total)

            if best_idx is not None and best_delta is not None:
                return int(best_idx), {
                    "is_fallback": 0,
                    "tries": int(evaluated),
                    "target_relation": -2,  # global mode
                    "found_relation": -2,
                }, best_delta, int(best_delta_total)

            # fall through to fallback below
            evaluated = 0

        # Threshold-based image similarity: use batched scoring for speed.
        state = self._build_global_state(cur_image_ids, cur_caption_ids)

        # 1) Sample candidate indices (keep original order; skip used).
        cand_indices: list[int] = []
        for _ in range(max(1, self.global_num_candidates)):
            ci = int(self._random_index(rng))
            if ci in used:
                continue
            cand_indices.append(ci)
        evaluated = int(len(cand_indices))

        if evaluated > 0:
            # 2) Gather candidate ids / embeddings in a batch.
            pairs = [self.dataset.pairs[int(ci)] for ci in cand_indices]
            cand_image_ids = [int(p.image_id) for p in pairs]
            cand_caption_ids = [int(p.caption_id) for p in pairs]
            C = int(len(cand_indices))
            k = int(len(state.image_ids))

            text_thr = float(self.oracle.cfg.text_sim_threshold)
            img_thr = float(self.oracle.cfg.image_sim_threshold)

            E_i = self._image_emb(cand_image_ids)  # [C, Di]
            E_t = self._caption_emb(cand_caption_ids)  # [C, Dt]

            # ---- side 1: (i_existing, t_new) ----
            # sim_img_1: [k, C]
            sim_img_1 = (state.img_emb @ E_i.T) >= img_thr

            # sim_txt_1: max over captions(i_existing) vs t_new, [k, C]
            if int(state.cmax) <= 0:
                sim_txt_1 = torch.zeros((k, C), dtype=torch.bool, device=self.compute_device)
            else:
                # sims: [k, cmax, C]
                sims = torch.einsum("kcd,od->kco", state.capset_emb, E_t)  # [k, cmax, C]
                sims = sims.masked_fill(~state.capset_mask.unsqueeze(-1), -1e9)
                sim_txt_1 = sims.max(dim=1).values >= text_thr  # [k, C]

            # ---- side 2: (i_new, t_existing) ----
            sim_img_2 = (state.cap_img_emb @ E_i.T) >= img_thr  # [k, C]

            # Build candidate caption-set tensor: [C, c2max, Dt] + mask
            rows_list = [self._caption_rows_for_image_dev(int(iid)) for iid in cand_image_ids]
            c2max = max((int(r.numel()) for r in rows_list), default=0)
            if c2max <= 0:
                sim_txt_2 = torch.zeros((k, C), dtype=torch.bool, device=self.compute_device)
            else:
                row_mat = torch.full((C, c2max), -1, dtype=torch.long, device=self.compute_device)
                mask = torch.zeros((C, c2max), dtype=torch.bool, device=self.compute_device)
                for ii, rr in enumerate(rows_list):
                    n = int(rr.numel())
                    if n > 0:
                        row_mat[ii, :n] = rr
                        mask[ii, :n] = True
                flat = row_mat.clamp_min(0).reshape(-1)
                cap_emb_all = self._teacher_caption_emb()
                if cap_emb_all.device != self.compute_device:
                    capset = cap_emb_all.index_select(0, flat.cpu()).to(self.compute_device).float().view(C, c2max, -1)
                else:
                    capset = cap_emb_all.index_select(0, flat).float().view(C, c2max, -1)  # [C, c2max, Dt]

                # sims2: [k, C, c2max] = cap_emb[k,d] dot capset[C,m,d]
                sims2 = torch.einsum("kd,cmd->kcm", state.cap_emb, capset)
                sims2 = sims2.masked_fill(~mask.unsqueeze(0), -1e9)
                sim_txt_2 = sims2.max(dim=2).values >= text_thr  # [k, C]

            # 3) Compute delta counts per candidate (shape [R, C]).
            rel_keys = sorted(self.mix.probs().keys())
            if self.relation_mode == "full":
                def _counts(sim_txt: torch.Tensor, sim_img: torch.Tensor) -> torch.Tensor:
                    # returns [4, C] counts in relation order (1..4)
                    r1 = (sim_txt & sim_img).sum(dim=0)
                    r2 = (sim_txt & (~sim_img)).sum(dim=0)
                    r3 = ((~sim_txt) & sim_img).sum(dim=0)
                    r4 = ((~sim_txt) & (~sim_img)).sum(dim=0)
                    return torch.stack([r1, r2, r3, r4], dim=0)  # [4, C]

                side1 = _counts(sim_txt_1, sim_img_1)
                side2 = _counts(sim_txt_2, sim_img_2)
                delta_counts = side1 + side2  # [4, C]
            elif self.relation_mode == "text_only":
                # relation 1: similar_text, relation 2: different_text
                s1 = sim_txt_1.sum(dim=0)
                s2 = sim_txt_2.sum(dim=0)
                delta_counts = torch.stack([s1 + s2, (k - s1) + (k - s2)], dim=0)  # [2, C]
            elif self.relation_mode == "image_only":
                s1 = sim_img_1.sum(dim=0)
                s2 = sim_img_2.sum(dim=0)
                delta_counts = torch.stack([s1 + s2, (k - s1) + (k - s2)], dim=0)  # [2, C]
            else:
                raise ValueError(f"Unknown relation_mode: {self.relation_mode!r}")

            delta_total = int(2 * k)

            # 4) Score each candidate (L1 distance to target mix after adding deltas).
            tgt_probs = self.mix.probs()
            tgt = torch.tensor([tgt_probs[int(r)] for r in rel_keys], device=self.compute_device, dtype=torch.float32).view(
                len(rel_keys), 1
            )
            cur = torch.tensor(
                [cur_counts.get(int(r), 0) for r in rel_keys],
                device=self.compute_device,
                dtype=torch.float32,
            ).view(len(rel_keys), 1)
            denom = float(max(1, int(cur_total + delta_total)))
            new_p = (cur + delta_counts.float()) / denom  # [R, C]
            scores = (new_p - tgt).abs().sum(dim=0)  # [C]

            best_pos = int(torch.argmin(scores).item())  # first min => matches sequential tie-break
            best_idx = int(cand_indices[best_pos])
            best_delta = {int(rel_keys[i]): int(delta_counts[i, best_pos].item()) for i in range(len(rel_keys))}
            best_delta_total = int(delta_total)
            return int(best_idx), {
                "is_fallback": 0,
                "tries": int(evaluated),
                "target_relation": -2,  # global mode
                "found_relation": -2,
            }, best_delta, int(best_delta_total)

        # Fallback: any unused random index
        for _ in range(self.max_tries):
            cand_idx = self._random_index(rng)
            if cand_idx not in used:
                # For fallback in global mode, any unused index is fine; compute delta exactly.
                if bool(self.oracle.cfg.use_image_topk):
                    delta, delta_total = self._global_delta(
                        cand_idx=int(cand_idx), cur_image_ids=cur_image_ids, cur_caption_ids=cur_caption_ids
                    )
                else:
                    delta, delta_total = self._global_delta_from_state(cand_idx=int(cand_idx), state=state)
                return int(cand_idx), {
                    "is_fallback": 1,
                    "tries": int(evaluated),
                    "target_relation": -2,
                    "found_relation": -2,
                }, delta, int(delta_total)

        cand_idx = self._random_index(rng)
        if bool(self.oracle.cfg.use_image_topk):
            delta, delta_total = self._global_delta(
                cand_idx=int(cand_idx), cur_image_ids=cur_image_ids, cur_caption_ids=cur_caption_ids
            )
        else:
            delta, delta_total = self._global_delta_from_state(cand_idx=int(cand_idx), state=state)
        return int(cand_idx), {
            "is_fallback": 1,
            "tries": int(evaluated),
            "target_relation": -2,
            "found_relation": -2,
        }, delta, int(delta_total)

    def __iter__(self) -> Iterator[list[Any]]:
        rng = random.Random(self.seed + self.epoch)

        num_batches = len(self)
        for _ in range(num_batches):
            mode = self.mode.lower().strip()

            # ---- Continuum experiment: block-wise global (K blocks) with a single global seed ----
            if mode in ("block_global", "block", "bg"):
                K = max(1, int(self.num_blocks))
                K = min(K, int(self.batch_size))
                base = int(self.batch_size) // K
                rem = int(self.batch_size) % K
                block_sizes = [(base + 1) if i < rem else base for i in range(K)]

                used: set[int] = set()
                batch: list[Any] = []
                rel_keys = sorted(self.mix.probs().keys())

                # Global seed (the ONLY random seed pair for the whole batch)
                seed_idx = int(self._random_index(rng))
                batch.append((seed_idx, {"is_fallback": 1, "tries": 0, "target_relation": -1, "found_relation": -1}))
                used.add(seed_idx)
                seed_image_id = int(self.dataset.pairs[seed_idx].image_id)

                for bi, bsz in enumerate(block_sizes):
                    bsz = int(bsz)
                    if bsz <= 0:
                        continue

                    # Block anchor:
                    # - block 0 uses the global seed directly
                    # - other blocks choose their anchor via SINGLE sampling w.r.t the global seed image
                    if bi == 0:
                        anchor_idx = seed_idx
                        anchor_meta = {"is_fallback": 1, "tries": 0, "target_relation": -1, "found_relation": -1}
                    else:
                        rel = self.mix.sample_relation(rng)
                        anchor_idx, anchor_meta = self._find_candidate_single(
                            seed_image_id, int(rel), rng=rng, used=used
                        )
                        batch.append((int(anchor_idx), anchor_meta))
                        used.add(int(anchor_idx))

                    # Local block state (global greedy within block)
                    cur_counts = {int(k): 0 for k in rel_keys}
                    cur_total = 0
                    cur_image_ids: list[int] = [int(self.dataset.pairs[int(anchor_idx)].image_id)]
                    cur_caption_ids: list[int] = [int(self.dataset.pairs[int(anchor_idx)].caption_id)]

                    # We already placed anchor for block>0; for block 0 the anchor is seed already in batch.
                    # Fill the remaining (bsz-1) items in this block via global greedy within-block.
                    while len(cur_image_ids) < int(bsz):
                        cand_idx, meta, delta, delta_total = self._find_candidate_global(
                            rng=rng,
                            used=used,
                            cur_counts=cur_counts,
                            cur_total=cur_total,
                            cur_image_ids=cur_image_ids,
                            cur_caption_ids=cur_caption_ids,
                        )
                        # Tag meta to distinguish block-global from full-batch global.
                        meta = dict(meta)
                        meta["target_relation"] = -3
                        meta["found_relation"] = -3

                        batch.append((int(cand_idx), meta))
                        used.add(int(cand_idx))

                        for r in cur_counts.keys():
                            cur_counts[int(r)] += int(delta.get(int(r), 0))
                        cur_total += int(delta_total)
                        cur_image_ids.append(int(self.dataset.pairs[int(cand_idx)].image_id))
                        cur_caption_ids.append(int(self.dataset.pairs[int(cand_idx)].caption_id))

                yield batch
                continue

            # ---- Existing modes: single / multi / global ----
            used = set()
            batch = []

            # Seed with a random positive pair
            idx0 = self._random_index(rng)
            batch.append((int(idx0), {"is_fallback": 1, "tries": 0, "target_relation": -1, "found_relation": -1}))
            used.add(int(idx0))

            rel_keys = sorted(self.mix.probs().keys())
            cur_counts = {int(k): 0 for k in rel_keys}
            cur_total = 0
            cur_image_ids: list[int] = [int(self.dataset.pairs[int(idx0)].image_id)]
            cur_caption_ids: list[int] = [int(self.dataset.pairs[int(idx0)].caption_id)]

            while len(batch) < self.batch_size:
                if mode in ("single", "c"):
                    anchor_item = batch[rng.randrange(0, len(batch))]
                    anchor_idx = int(anchor_item[0]) if isinstance(anchor_item, tuple) else int(anchor_item)
                    anchor_image_id = int(self.dataset.pairs[int(anchor_idx)].image_id)
                    rel = self.mix.sample_relation(rng)
                    cand_idx, meta = self._find_candidate_single(anchor_image_id, rel, rng=rng, used=used)
                elif mode in ("multi", "b"):
                    rel = self.mix.sample_relation(rng)
                    k = min(max(1, self.num_anchors), len(batch))
                    anchor_items = [batch[rng.randrange(0, len(batch))] for _ in range(k)]
                    anchor_idxs = [int(x[0]) if isinstance(x, tuple) else int(x) for x in anchor_items]
                    anchor_image_ids = [int(self.dataset.pairs[int(a)].image_id) for a in anchor_idxs]
                    cand_idx, meta = self._find_candidate_multi(anchor_image_ids, rel, rng=rng, used=used)
                elif mode in ("global", "a"):
                    cand_idx, meta, delta, delta_total = self._find_candidate_global(
                        rng=rng,
                        used=used,
                        cur_counts=cur_counts,
                        cur_total=cur_total,
                        cur_image_ids=cur_image_ids,
                        cur_caption_ids=cur_caption_ids,
                    )
                    for r in cur_counts.keys():
                        cur_counts[int(r)] += int(delta.get(int(r), 0))
                    cur_total += int(delta_total)
                else:
                    raise ValueError(
                        f"Unknown semantic sampler mode: {self.mode!r} (expected single|multi|global|block_global)"
                    )

                batch.append((int(cand_idx), meta))
                used.add(int(cand_idx))

                if mode in ("global", "a"):
                    cur_image_ids.append(int(self.dataset.pairs[int(cand_idx)].image_id))
                    cur_caption_ids.append(int(self.dataset.pairs[int(cand_idx)].caption_id))

            yield batch

