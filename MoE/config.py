from dataclasses import dataclass, field
from typing import Dict


@dataclass
class RetrievalConfig:
    """Number of candidates retrieved from each source before union."""
    top_seq: int = 20
    top_gcn: int = 20
    top_sem: int = 20
    top_M:   int = 20
    top_K:   int = 5


@dataclass
class GatingConfig:
    """Config for the MLP gating network.
    Context features (mode='context', input_dim=7):
      0: norm_seq_len         — len_seq / max_seq_len
      1: agree_gcn            — Spearman rank corr seq vs gcn, mapped [0,1]
      2: agree_sem            — Spearman rank corr seq vs sem, mapped [0,1]
      3: agree_gcn_sem        — Spearman rank corr gcn vs sem, mapped [0,1]  
      4: seq_confidence       — max - mean of SASRec scores  ∈ [0,1]         
      5: gcn_confidence       — max - mean of GCN scores     ∈ [0,1]
      6: sem_confidence       — max - mean of Sem scores     ∈ [0,1]
    """
    input_dim:    int   = 7         
    hidden_dims:  list  = field(default_factory=lambda: [32, 16])
    dropout:      float = 0.2
    lr:           float = 1e-3
    epochs:       int   = 50
    batch_size:   int   = 256
    weight_decay: float = 1e-4

    default_weights: list = field(
        default_factory=lambda: [1/3, 1/3, 1/3]
    )

    gating_mode:   str   = 'context'   
    entropy_reg_weight: float = 0.05
    expert_quality_threshold: float = 0.1
    concentration_weight: float = 0.02
    max_seq_len:       int   = 50
    cold_threshold:    int   = 5

    use_seq_len_in_gating: bool = False


@dataclass
class ScoreConfig:
    """
    s1(u,i) = alpha * s0(u,i) + (1 - alpha) * s_rerank(u,i)
    """
    alpha: float = 0.5

    dataset_alpha: Dict[str, float] = field(default_factory=lambda: {
        'yelp':               0.5,
        'amazon':             0.5,
        'amazon_musical':     0.5,
        'amazon_industrial':  0.5,
        'goodreads':          0.5,
    })
    use_adaptive_alpha: bool = False


@dataclass
class MoEConfig:
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    gating:    GatingConfig    = field(default_factory=GatingConfig)
    scoring:   ScoreConfig     = field(default_factory=ScoreConfig)

    gating_model_path: str = None

    use_seq:        bool = True
    use_gcn:        bool = True
    use_semantic:   bool = True
    use_reranker:   bool = True
    use_user_agent: bool = True

DEFAULT_CONFIG = MoEConfig()


def get_config_for_dataset(dataset: str = None) -> MoEConfig:
    """
    Return a MoEConfig tuned for each dataset.

    Amazon:    GCN is strong → balanced
    Yelp:      SASRec is strong → seq heavy, Semantic weak → suppress
    Goodreads: both GCN and SASRec fail due to sparsity →
               FAISS-driven entirely
    """
    cfg = MoEConfig()

    cfg.retrieval.top_seq = 20
    cfg.retrieval.top_gcn = 20
    cfg.retrieval.top_sem = 20
    cfg.retrieval.top_M   = 20
    cfg.retrieval.top_K   = 5
    cfg.gating.default_weights = [1/3, 1/3, 1/3]

    return cfg