from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple


@dataclass(frozen=True)
class CvclVocab:
    word2idx: Dict[str, int]
    idx2word: Dict[int, str]

    @property
    def pad_id(self) -> int:
        return int(self.word2idx.get("<pad>", 0))

    @property
    def unk_id(self) -> int:
        return int(self.word2idx.get("<unk>", 1))


def _iter_captions(dataset: Any) -> Iterable[str]:
    # dataset is expected to expose `.pairs` each with `.caption`, for COCO/SayCam datasets here
    pairs = getattr(dataset, "pairs", None)
    if pairs is None:
        raise ValueError("Dataset must have a .pairs property for CVCL vocab building.")
    for p in pairs:
        yield str(getattr(p, "caption"))


def build_cvcl_vocab_from_dataset(
    dataset: Any,
    *,
    min_freq: int = 1,
    max_vocab: int = 50000,
) -> CvclVocab:
    """
    Build a whitespace-tokenized vocab aligned with CVCL default text encoder behavior.
    """
    counter: Counter[str] = Counter()
    for cap in _iter_captions(dataset):
        toks = [t for t in cap.strip().split() if t]
        counter.update(toks)

    # Special tokens: keep indices stable
    word2idx: Dict[str, int] = {"<pad>": 0, "<unk>": 1}

    # Most common tokens
    for w, c in counter.most_common():
        if c < int(min_freq):
            continue
        if w in word2idx:
            continue
        word2idx[w] = len(word2idx)
        if len(word2idx) >= int(max_vocab):
            break

    idx2word = {i: w for w, i in word2idx.items()}
    return CvclVocab(word2idx=word2idx, idx2word=idx2word)


def cvcl_tokenize_batch(
    texts: List[str], vocab: CvclVocab
) -> Tuple[List[List[int]], List[int]]:
    token_lists: List[List[int]] = []
    lengths: List[int] = []
    for s in texts:
        toks = [t for t in str(s).strip().split() if t]
        ids = [vocab.word2idx.get(t, vocab.unk_id) for t in toks]
        if not ids:
            ids = [vocab.unk_id]
        token_lists.append(ids)
        lengths.append(len(ids))
    return token_lists, lengths

