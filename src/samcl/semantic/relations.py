from __future__ import annotations

from dataclasses import dataclass

import torch

from samcl.data.coco_pairs import CocoPairsDataset
from samcl.data.saycam_pairs import SayCamPairsDataset
from samcl.data.wds_pairs import WdsPairsDataset
from samcl.teachers.cache import TeacherEmbeddingCache


@dataclass(frozen=True)
class SemanticRelationConfig:
    # Decide similar vs different for text(i, caption):
    # sim_text(i, t) = max_{c in captions(i)} cos(e_t, e_c)
    text_sim_threshold: float = 0.55

    # Decide similar vs different for images(i, j):
    # sim_img(i, j) = cos(E_i, E_j)
    image_sim_threshold: float = 0.70

    # Optional: use top-k neighbor list instead of threshold for image similarity
    use_image_topk: bool = False
    image_topk: int = 50

    # Relation mode:
    #   - "full": 4-way relation (text_sim x image_sim) in {1,2,3,4}
    #   - "text_only": 2-way relation based only on text similarity in {1,2}
    #   - "image_only": 2-way relation based only on image similarity in {1,2}
    relation_mode: str = "full"


class ImageTopKCache:
    """
    Lazy top-k neighbor cache for image-image similarity on teacher embeddings.
    """

    def __init__(self, cache: TeacherEmbeddingCache, *, topk: int) -> None:
        assert cache.image_emb is not None and cache.image_id_to_row is not None
        self.cache = cache
        self.topk = int(topk)
        self._neighbors: dict[int, torch.Tensor] = {}

        # Keep on CPU for cheaper all-vs-one dot products in this minimal version.
        self._emb = cache.image_emb.float().contiguous()

    def neighbors(self, image_id: int) -> torch.Tensor:
        image_id = int(image_id)
        if image_id in self._neighbors:
            return self._neighbors[image_id]

        q = self.cache.get_image_emb(image_id).float().view(1, -1)  # [1, D]
        sims = (q @ self._emb.T).squeeze(0)  # [N]

        # remove self if present in the same embedding set
        row = self.cache.image_id_to_row[image_id]
        sims[row] = -1e9

        k = min(self.topk, sims.numel())
        vals, idx = torch.topk(sims, k=k, largest=True)
        nn_image_ids = self.cache.image_ids[idx].clone()  # [k]
        self._neighbors[image_id] = nn_image_ids
        return nn_image_ids


class SemanticRelationOracle:
    """
    Core research abstraction:
      get_semantic_relation(image_id, caption_id) -> {1,2,3,4}

    Teachers are frozen and only used here + in sampler decisions.
    """

    def __init__(
        self,
        *,
        dataset: CocoPairsDataset | SayCamPairsDataset | WdsPairsDataset,
        teacher_cache: TeacherEmbeddingCache,
        cfg: SemanticRelationConfig,
    ) -> None:
        self.dataset = dataset
        self.teacher_cache = teacher_cache
        self.cfg = cfg
        self._img_topk = (
            ImageTopKCache(teacher_cache, topk=cfg.image_topk) if cfg.use_image_topk else None
        )

    def text_similarity(self, image_id: int, caption_id: int) -> float:
        """
        sim_text(i, t) = max_{c in captions(i)} cos(e_t, e_c)
        """
        cap_ids = self.dataset.caption_ids_for_image(int(image_id))
        if not cap_ids:
            return 0.0

        e_t = self.teacher_cache.get_caption_emb(int(caption_id)).float()
        e_cs = torch.stack([self.teacher_cache.get_caption_emb(int(cid)).float() for cid in cap_ids], dim=0)
        sims = (e_cs @ e_t)  # [num_caps]
        return float(torch.max(sims).item())

    def image_similarity(self, image_id_a: int, image_id_b: int) -> float:
        e_a = self.teacher_cache.get_image_emb(int(image_id_a)).float()
        e_b = self.teacher_cache.get_image_emb(int(image_id_b)).float()
        return float(torch.dot(e_a, e_b).item())

    def is_similar_text(self, image_id: int, caption_id: int) -> bool:
        return self.text_similarity(image_id, caption_id) >= float(self.cfg.text_sim_threshold)

    def is_similar_image(self, image_id_a: int, image_id_b: int) -> bool:
        if self._img_topk is not None:
            nn = self._img_topk.neighbors(image_id_a)
            return bool((nn == int(image_id_b)).any().item())
        return self.image_similarity(image_id_a, image_id_b) >= float(self.cfg.image_sim_threshold)

    def get_semantic_relation(self, image_id: int, caption_id: int) -> int:
        """
        Relation types:
          1: similar text, similar image
          2: similar text, different image
          3: different text, similar image
          4: different text, different image
        """
        image_id = int(image_id)
        caption_id = int(caption_id)

        caption_image_id = self.dataset.image_id_for_caption(caption_id)
        sim_text = self.is_similar_text(image_id, caption_id)
        sim_img = self.is_similar_image(image_id, caption_image_id)

        if sim_text and sim_img:
            return 1
        if sim_text and not sim_img:
            return 2
        if (not sim_text) and sim_img:
            return 3
        return 4

    def get_relation(self, image_id: int, caption_id: int, *, mode: str | None = None) -> int:
        """
        Generalized relation getter supporting 4-way (full) and 2-way (text/image only).

        Returns:
          - mode="full": {1,2,3,4}
          - mode="text_only": {1(similar_text), 2(different_text)}
          - mode="image_only": {1(similar_image), 2(different_image)}
        """
        m = (mode or self.cfg.relation_mode or "full").lower().strip()
        image_id = int(image_id)
        caption_id = int(caption_id)

        if m == "full":
            return int(self.get_semantic_relation(image_id, caption_id))

        caption_image_id = self.dataset.image_id_for_caption(caption_id)
        if m == "text_only":
            return 1 if self.is_similar_text(image_id, caption_id) else 2
        if m == "image_only":
            return 1 if self.is_similar_image(image_id, caption_image_id) else 2
        raise ValueError(f"Unknown relation mode: {mode!r} (use full|text_only|image_only)")

    def batch_cross_relation_histogram(
        self, image_ids: torch.Tensor, caption_ids: torch.Tensor
    ) -> dict[int, int]:
        """
        Compute histogram over relations for all cross pairs (i_a, t_b) where a != b.
        Returns counts keyed by {1,2,3,4}.
        """
        return self.batch_cross_relation_histogram_mode(image_ids, caption_ids, mode="full")

    def batch_cross_relation_histogram_mode(
        self, image_ids: torch.Tensor, caption_ids: torch.Tensor, *, mode: str | None = None
    ) -> dict[int, int]:
        """
        Compute histogram over relations for all cross pairs (i_a, t_b) where a != b.
        """
        m = (mode or self.cfg.relation_mode or "full").lower().strip()
        b = int(image_ids.numel())
        if m == "full":
            counts: dict[int, int] = {1: 0, 2: 0, 3: 0, 4: 0}
        else:
            counts = {1: 0, 2: 0}
        for a in range(b):
            i_a = int(image_ids[a].item())
            for bb in range(b):
                if bb == a:
                    continue
                t_b = int(caption_ids[bb].item())
                r = int(self.get_relation(i_a, t_b, mode=m))
                counts[r] += 1
        return counts

