
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional

current_dir = os.path.dirname(os.path.abspath(__file__))
moe_root    = os.path.dirname(current_dir)
if moe_root not in sys.path:
    sys.path.insert(0, moe_root)

from config import GatingConfig, DEFAULT_CONFIG


class GatingMLP(nn.Module):

    def __init__(self, cfg: GatingConfig = None):
        super().__init__()
        cfg = cfg or DEFAULT_CONFIG.gating

        layers = []
        in_dim = cfg.input_dim
        for h_dim in cfg.hidden_dims:
            layers += [
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.ReLU(),
                nn.Dropout(cfg.dropout),
            ]
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, 3))   
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        return F.softmax(logits, dim=-1)      


def _spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return 0.0
    if np.all(a == a[0]) or np.all(b == b[0]):
        return 0.0
    from scipy.stats import spearmanr
    corr, _ = spearmanr(a, b)
    return float(corr) if not np.isnan(corr) else 0.0


def _expert_confidence(scores: Dict[str, float]) -> float:
    if not scores:
        return 0.0
    vals = np.array(list(scores.values()), dtype=np.float32)
    return float(vals.max() - vals.mean())



def extract_context_features(
    seq:        List[int],
    len_seq:    int,
    seq_scores: Dict[str, float],      
    gcn_scores: Dict[str, float],      
    sem_scores: Dict[str, float],       
    gcn_norm:   Optional[torch.Tensor], 
    cfg:        GatingConfig,
) -> List[float]:
    max_seq = getattr(cfg, 'max_seq_len', 50)

    norm_seq_len = min(len_seq / max(max_seq, 1), 1.0)

    agree_gcn = 0.0
    if seq_scores and gcn_scores:
        common = set(seq_scores) & set(gcn_scores)
        if len(common) >= 3:
            s_seq = np.array([seq_scores[n] for n in common], dtype=np.float32)
            s_gcn = np.array([gcn_scores[n] for n in common], dtype=np.float32)
            agree_gcn = (_spearman_corr(s_seq, s_gcn) + 1.0) / 2.0  # [-1,1] → [0,1]

    agree_sem = 0.0
    if seq_scores and sem_scores:
        common = set(seq_scores) & set(sem_scores)
        if len(common) >= 3:
            s_seq = np.array([seq_scores[n] for n in common], dtype=np.float32)
            s_sem = np.array([sem_scores[n] for n in common], dtype=np.float32)
            agree_sem = (_spearman_corr(s_seq, s_sem) + 1.0) / 2.0

    # Direct consensus between the 2 non-seq experts.
    # High → gcn and sem agree → gating can boost both.
    # Low  → experts conflict → seq acts as natural tiebreaker.
    agree_gcn_sem = 0.0
    if gcn_scores and sem_scores:
        common = set(gcn_scores) & set(sem_scores)
        if len(common) >= 3:
            s_gcn = np.array([gcn_scores[n] for n in common], dtype=np.float32)
            s_sem = np.array([sem_scores[n] for n in common], dtype=np.float32)
            agree_gcn_sem = (_spearman_corr(s_gcn, s_sem) + 1.0) / 2.0

    seq_confidence = _expert_confidence(seq_scores)
    gcn_confidence = _expert_confidence(gcn_scores)
    sem_confidence = _expert_confidence(sem_scores)

    return [
        norm_seq_len,   # 0
        agree_gcn,      # 1
        agree_sem,      # 2
        agree_gcn_sem,  # 3
        seq_confidence, # 4 
        gcn_confidence, # 5
        sem_confidence, # 6
    ]

class GatingNetwork:
    def __init__(
        self,
        cfg:        GatingConfig  = None,
        model_path: str           = None,
        device:     torch.device  = None,
    ):
        self.cfg    = cfg or DEFAULT_CONFIG.gating
        self.device = device or torch.device('cpu')
        self.model  = GatingMLP(self.cfg).to(self.device)
        self.trained = False

        self.feat_mean: Optional[np.ndarray] = None
        self.feat_std:  Optional[np.ndarray] = None

        if model_path and os.path.exists(model_path):
            self.load(model_path)
        else:
            print(f'[GatingNetwork] No checkpoint — using default weights {self.cfg.default_weights}')

    def predict_from_context(
        self,
        context_features: List[float],
    ) -> Tuple[float, float, float]:
        """
        Predict gate weights from user-context feature vector.

        Returns:
            (g_seq, g_gcn, g_sem) — sum = 1
        """
        feat = np.array(context_features, dtype=np.float32)

        if not self.trained:
            dw = self.cfg.default_weights
            return float(dw[0]), float(dw[1]), float(dw[2])

        if self.feat_mean is not None and self.feat_std is not None:
            feat = (feat - self.feat_mean) / (self.feat_std + 1e-8)

        x = torch.tensor(feat, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.model.eval()
        with torch.no_grad():
            w = self.model(x).cpu().numpy()[0]

        return float(w[0]), float(w[1]), float(w[2])


    def predict(
        self,
        signal_scores: Dict[str, Dict[str, float]],
        len_seq: int = 0,
    ) -> Dict[str, Tuple[float, float, float]]:
        if not signal_scores:
            return {}

        items = list(signal_scores.keys())

        if not self.trained:
            if len_seq <= self.cfg.cold_threshold:
                dw = [0.15, 0.35, 0.50]   # cold: trust semantic more
            else:
                dw = self.cfg.default_weights
            return {it: (float(dw[0]), float(dw[1]), float(dw[2])) for it in items}

        def _minmax(d: Dict[str, float]) -> Dict[str, float]:
            if not d: return {}
            vals = np.array(list(d.values()))
            lo, hi = vals.min(), vals.max()
            if hi == lo: return {k: 0.5 for k in d}
            return {k: float((v - lo) / (hi - lo)) for k, v in d.items()}

        raw_seq = {it: scores.get('seq', 0.0) for it, scores in signal_scores.items()}
        raw_gcn = {it: scores.get('gcn', 0.0) for it, scores in signal_scores.items()}
        raw_sem = {it: scores.get('sem', 0.0) for it, scores in signal_scores.items()}

        norm_seq = _minmax(raw_seq)
        norm_gcn = _minmax(raw_gcn)
        norm_sem = _minmax(raw_sem)

        ctx = extract_context_features(
            seq=[],
            len_seq=len_seq,
            seq_scores=norm_seq,
            gcn_scores=norm_gcn,
            sem_scores=norm_sem,
            gcn_norm=None,
            cfg=self.cfg,
        )

        g_seq, g_gcn, g_sem = self.predict_from_context(ctx)
        return {it: (g_seq, g_gcn, g_sem) for it in items}


    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'cfg':              self.cfg,
            'norm_mean':        self.feat_mean.tolist() if self.feat_mean is not None else None,
            'norm_std':         self.feat_std.tolist()  if self.feat_std  is not None else None,
            'gating_mode':      getattr(self.cfg, 'gating_mode', 'context'),
            'feature_version':  'v2.2',
            'feature_names':    [
                'norm_seq_len',
                'agree_gcn',
                'agree_sem',
                'agree_gcn_sem',   
                'seq_confidence',  
                'gcn_confidence',
                'sem_confidence',
            ],
        }, path)
        print(f'[GatingNetwork] Saved → {path}')

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        if 'cfg' in ckpt:
            self.cfg   = ckpt['cfg']
            self.model = GatingMLP(self.cfg).to(self.device)
            mode    = getattr(self.cfg, 'gating_mode', 'context')
            fv      = ckpt.get('feature_version', 'v2.0')
            fnames  = ckpt.get('feature_names', [])
            print(f'[GatingNetwork] Loaded: mode={mode}, input_dim={self.cfg.input_dim}, '
                  f'feature_version={fv}')
            if fnames:
                print(f'[GatingNetwork] Features: {fnames}')

        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval()
        self.trained = True

        if ckpt.get('norm_mean') is not None:
            self.feat_mean = np.array(ckpt['norm_mean'], dtype=np.float32)
            self.feat_std  = np.array(ckpt['norm_std'],  dtype=np.float32)
            print(f'[GatingNetwork] Norm params: mean={self.feat_mean.round(3)}')
