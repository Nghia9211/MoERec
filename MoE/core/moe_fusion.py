import os
import sys
import numpy as np
from typing import Dict, List, Tuple, Optional

current_dir = os.path.dirname(os.path.abspath(__file__))
moe_root    = os.path.dirname(current_dir)   
if moe_root not in sys.path:
    sys.path.insert(0, moe_root)

from config              import MoEConfig, DEFAULT_CONFIG
from gating.gating_network import GatingNetwork

def _minmax_normalize(score_dict: Dict[str, float]) -> Dict[str, float]:
    if not score_dict: return {}
    vals = np.array(list(score_dict.values()))
    min_v, max_v = np.min(vals), np.max(vals)
    if max_v == min_v:
        return {k: 0.5 for k in score_dict}
    return {k: float((v - min_v) / (max_v - min_v)) for k, v in score_dict.items()}

def _extract_signal(signal_scores: Dict[str, Dict[str, float]], signal_key: str) -> Dict[str, float]:
    return {name: scores.get(signal_key, 0.0) for name, scores in signal_scores.items()}

class MoEFusion:
    def __init__(self, gating: GatingNetwork, cfg: MoEConfig = None):
        self.gating = gating
        self.cfg    = cfg or DEFAULT_CONFIG

    def fuse(
        self,
        signal_scores: Dict[str, Dict[str, float]],
        len_seq: int = 0,
        top_m: int = None,
        debug: bool = False,
    ) -> Tuple[List[str], Dict[str, float], dict]:
        
        top_m = top_m or self.cfg.retrieval.top_M
        if not signal_scores: return [], {}, {}

        raw_seq = _extract_signal(signal_scores, 'seq')
        raw_gcn = _extract_signal(signal_scores, 'gcn')
        raw_sem = _extract_signal(signal_scores, 'sem')

        norm_seq = _minmax_normalize(raw_seq)
        norm_gcn = _minmax_normalize(raw_gcn)
        norm_sem = _minmax_normalize(raw_sem)

        # Pass raw signal_scores to gating.predict(); normalization is handled internally.
        # Fusion uses the pre-normalized norm_seq/gcn/sem computed above.
        gate_weights = self.gating.predict(signal_scores, len_seq=len_seq)

        fused_scores: Dict[str, float] = {}
        actual_gates = []
        for name in signal_scores:
            g1, g2, g3 = gate_weights.get(name, (1/3, 1/3, 1/3))
            s_seq = norm_seq.get(name, 0.0)
            s_gcn = norm_gcn.get(name, 0.0)
            s_sem = norm_sem.get(name, 0.0)

            if not self.cfg.use_seq: s_seq, g1 = 0.0, 0.0
            if not self.cfg.use_gcn: s_gcn, g2 = 0.0, 0.0
            if not self.cfg.use_semantic: s_sem, g3 = 0.0, 0.0

            total_g = g1 + g2 + g3
            if total_g > 0:
                g1, g2, g3 = g1/total_g, g2/total_g, g3/total_g
            
            actual_gates.append((g1, g2, g3))

            fused_scores[name] = g1 * s_seq + g2 * s_gcn + g3 * s_sem

        c_m = sorted(fused_scores, key=fused_scores.get, reverse=True)[:top_m]

        debug_info = {}
        if debug:
            avg_g1 = np.mean([g[0] for g in actual_gates]) if actual_gates else 0.0
            avg_g2 = np.mean([g[1] for g in actual_gates]) if actual_gates else 0.0
            avg_g3 = np.mean([g[2] for g in actual_gates]) if actual_gates else 0.0

            debug_info = {
                'avg_gates':    {'seq': round(avg_g1, 3), 'gcn': round(avg_g2, 3), 'sem': round(avg_g3, 3)},
                'n_candidates': len(signal_scores),
                'top_m':        top_m,
                'top_item':     c_m[0] if c_m else None,
                'top_score':    round(fused_scores.get(c_m[0], 0), 4) if c_m else 0,
                'trained_gating': self.gating.trained,
            }

        return c_m, fused_scores, debug_info

    def fuse_from_data(
        self,
        data: dict,
        signal_scores: Dict[str, Dict[str, float]],
        top_m: int = None,
        debug: bool = True,
    ) -> Tuple[List[str], Dict[str, float], dict]:
        
        len_seq = data.get('len_seq', 0)
        if len_seq == 0:
            raw_seq = data.get('seq', data.get('history', []))
            if isinstance(raw_seq, str):
                len_seq = len([x for x in raw_seq.split(',') if x.strip()])
            elif isinstance(raw_seq, (list, tuple)):
                len_seq = len(raw_seq)

        top_m = top_m or self.cfg.retrieval.top_M
        return self.fuse(signal_scores, len_seq=len_seq, top_m=top_m, debug=debug)