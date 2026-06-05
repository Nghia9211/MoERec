import os
import sys
import numpy as np
import math
from typing import Dict, List, Tuple

current_dir = os.path.dirname(os.path.abspath(__file__))
moe_root    = os.path.dirname(current_dir)
if moe_root not in sys.path:
    sys.path.insert(0, moe_root)

from config import ScoreConfig, DEFAULT_CONFIG


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sasrec_confidence(fused_scores: Dict[str, float]) -> float:
    if len(fused_scores) < 2:
        return 1.0
    sorted_vals = sorted(fused_scores.values(), reverse=True)
    top1, top2  = sorted_vals[0], sorted_vals[1]
    score_range = sorted_vals[0] - sorted_vals[-1]
    if score_range == 0:
        return 0.5
    margin = (top1 - top2) / score_range
    return float(np.clip(margin, 0.0, 1.0))

def _scores_to_rank_scores(scores_dict: Dict[str, float]) -> Dict[str, float]:

    if not scores_dict:
        return {}

    sorted_items = sorted(scores_dict, key=scores_dict.get, reverse=True)

    rank_scores = {}
    for rank, item in enumerate(sorted_items):
        rank_scores[item] = 1.0 / math.log2(rank + 2)

    return rank_scores


def moe_confidence_score(fused_scores: Dict[str, float]) -> float:
    if len(fused_scores) < 2:
        return 1.0

    sorted_vals = sorted(fused_scores.values(), reverse=True)
    score_range = sorted_vals[0] - sorted_vals[-1]

    if score_range == 0:
        return 0.0

    margin = (sorted_vals[0] - sorted_vals[1]) / score_range

    top3_mass = sum(sorted_vals[:min(3, len(sorted_vals))])
    total_mass = sum(sorted_vals)
    concentration = top3_mass / total_mass if total_mass > 0 else 0.5

    confidence = 0.6 * margin + 0.4 * concentration
    return float(np.clip(confidence, 0.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# ScoreCombiner
# ─────────────────────────────────────────────────────────────────────────────

class ScoreCombiner:

    EPOCH_ALPHA_BOOST: float = 0.0

    def __init__(self, cfg: ScoreConfig = None):
        self.cfg = cfg or DEFAULT_CONFIG.scoring

    # ─────────────────────────────────────────────────────────────────────
    # Alpha selection
    # ─────────────────────────────────────────────────────────────────────
    def get_alpha(
        self,
        dataset:       str,
        len_seq:       int,
        fused_scores:  Dict[str, float] = None,
        epoch:         int = 1,
    ) -> float:
        if not self.cfg.use_adaptive_alpha:
            base = self.cfg.alpha
            epoch_boost = (epoch - 1) * self.EPOCH_ALPHA_BOOST
            return float(np.clip(base + epoch_boost, 0.1, 0.9))

        alpha = self.cfg.dataset_alpha.get(dataset, self.cfg.alpha)

        epoch_boost = (epoch - 1) * self.EPOCH_ALPHA_BOOST
        alpha      += epoch_boost

        final_alpha = float(np.clip(alpha, 0.1, 0.9))
        if epoch > 1:
            print(f"[ScoreCombiner] epoch={epoch} alpha_boost={epoch_boost:.2f} "
                  f"final_alpha={final_alpha:.3f}")
        return final_alpha

    # ─────────────────────────────────────────────────────────────────────
    # Core combine
    # ─────────────────────────────────────────────────────────────────────

    def combine(
        self,
        fused_scores:   Dict[str, float],
        rerank_scores:  Dict[str, float],
        dataset:        str  = 'amazon',
        len_seq:        int  = 0,
        top_k:          int  = None,
        epoch:          int  = 1,
    ) -> Tuple[List[str], Dict[str, float], dict]:
        top_k = top_k or DEFAULT_CONFIG.retrieval.top_K

        if not fused_scores:
            c_k = sorted(rerank_scores, key=rerank_scores.get, reverse=True)[:top_k]
            return c_k, rerank_scores, {'alpha': 0.0, 'fallback': True}

        # Compute alpha before converting to ranks, so confidence uses raw s0 margin
        alpha = self.get_alpha(dataset, len_seq, fused_scores, epoch=epoch)
        beta  = 1.0 - alpha

        rank_s0 = _scores_to_rank_scores(fused_scores)

        # s1 = alpha * rank_s0 + beta * s_rerank
        all_items = set(rank_s0.keys()) | set(rerank_scores.keys())
        s1_scores: Dict[str, float] = {}

        for item in all_items:
            s0_val    = rank_s0.get(item, 0.0)
            s_rer_val = rerank_scores.get(item, 0.0)
            s1_scores[item] = alpha * s0_val + beta * s_rer_val

        c_k = sorted(s1_scores, key=s1_scores.get, reverse=True)[:top_k]

        debug_info = {
            'alpha':   round(alpha, 3),
            'beta':    round(beta, 3),
            'dataset': dataset,
            'len_seq': len_seq,
            'epoch':   epoch,
            'top_k':   top_k,
            'n_items': len(all_items),
            'top_item': c_k[0] if c_k else None,
        }

        return c_k, s1_scores, debug_info

    def combine_from_pipeline(
        self,
        fused_scores:  Dict[str, float],
        rerank_scores: Dict[str, float],
        data:          dict,
        args,
        top_k:         int = None,
        epoch:         int = 1,
    ) -> Tuple[List[str], Dict[str, float], dict]:
        dataset = (data.get('dataset') or
                   next((d for d in ['yelp', 'amazon', 'goodreads']
                         if d in getattr(args, 'data_dir', '')),
                        'amazon'))
        len_seq = data.get('len_seq', 0)
        top_k   = top_k or DEFAULT_CONFIG.retrieval.top_K

        return self.combine(
            fused_scores  = fused_scores,
            rerank_scores = rerank_scores,
            dataset       = dataset,
            len_seq       = len_seq,
            top_k         = top_k,
            epoch         = epoch,
        )