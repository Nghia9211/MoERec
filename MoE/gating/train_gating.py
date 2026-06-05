import os, sys, argparse, math, numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

current_dir  = os.path.dirname(os.path.abspath(__file__))  # MoE/gating/
moe_root     = os.path.dirname(current_dir)               # MoE/
project_root = os.path.dirname(moe_root)                  # moe_early_fusion/
for p in [moe_root, project_root, os.path.join(project_root, 'baseline')]:
    if p not in sys.path: sys.path.insert(0, p)

from config                        import MoEConfig, GatingConfig, DEFAULT_CONFIG
from gating.gating_network         import GatingMLP, GatingNetwork, extract_context_features
from experts.seq_scorer            import SeqScorer
from experts.gcn_scorer            import GCNScorer
from experts.semantic_scorer       import SemanticScorer
from dataset.general_dataset       import GeneralDataset

FEATURE_LABELS = [
    'norm_seq_len',
    'agree_gcn',
    'agree_sem',
    'agree_gcn_sem',   
    'seq_confidence', 
    'gcn_confidence',  
    'sem_confidence',  
]

DATASET_QUERY_PREFIX = {
    'goodreads': 'Books similar to: ',
    'yelp':      'Places and restaurants similar to: ',
    'amazon':    'User interested in: ',
}


def _minmax(d: Dict[str, float]) -> Dict[str, float]:
    if not d: return {}
    vals = np.array(list(d.values()))
    lo, hi = vals.min(), vals.max()
    if hi == lo: return {k: 0.5 for k in d}
    return {k: float((v - lo) / (hi - lo)) for k, v in d.items()}


def _ndcg_quality(gt_name: str, scores: Dict[str, float]) -> float:
    if not scores or gt_name not in scores: return 0.0
    rank = sorted(scores, key=scores.get, reverse=True).index(gt_name) + 1
    return 1.0 / math.log2(rank + 1)


def _quality_proportional_target(
    q_arr: np.ndarray,
    min_quality: float = 0.1,
) -> Optional[np.ndarray]:
    q_masked = q_arr.copy()
    q_masked[q_masked < min_quality] = 0.0
    total = q_masked.sum()
    if total < 1e-8:
        return None  
    return (q_masked / total).astype(np.float32)


def print_feature_report(X: np.ndarray):
    print(f"\n{'Signal':<18} | {'Mean':>7} | {'Std':>7} | {'%NonZero':>9}")
    print('-' * 54)
    for i in range(X.shape[1]):
        col = X[:, i]
        nz  = (np.abs(col) > 1e-9).mean() * 100
        lbl = FEATURE_LABELS[i] if i < len(FEATURE_LABELS) else f'f{i}'
        print(f"{lbl:<18} | {col.mean():>7.4f} | {col.std():>7.4f} | {nz:>8.1f}%")


def normalize_features(X: np.ndarray):
    mu  = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    return (X - mu) / std, mu, std


def compute_batch_scores(
    batch_seqs,
    batch_lens,
    batch_pools,
    batch_queries,      
    batch_user_ids,      
    seq_scorer,
    gcn_scorer,
    sem_scorer,
    id2name,
    device,
):
    B = len(batch_seqs)

    with torch.no_grad():
        all_logits = seq_scorer.model.forward_eval(batch_seqs, batch_lens.cpu().numpy())

    all_gcn_logits = None
    if gcn_scorer:
        node_embs = getattr(gcn_scorer, 'node_embs', {})  
        for i in range(B):
         
            raw_uid = batch_user_ids[i] if batch_user_ids is not None else None
            if raw_uid is not None:
                uid = str(raw_uid.item()) if hasattr(raw_uid, 'item') else str(raw_uid)
            else:
                uid = None
            h_u = node_embs.get(uid) if uid is not None else None
            if h_u is None:
                h_u = torch.zeros(gcn_scorer.gcn_norm.shape[1], device=device)
            else:
                h_u = F.normalize(h_u.to(device).float(), dim=0)
            h_users.append(h_u)
        h_users = torch.stack(h_users)                    
        all_gcn_logits = h_users @ gcn_scorer.gcn_norm.T   

    sem_map = [{} for _ in range(B)]
    if sem_scorer and sem_scorer.embedding_function:
        all_unique = set()
        for pool in batch_pools:
            all_unique.update(id2name.get(cid, f'item_{cid}') for cid in pool)
        unique_list  = list(all_unique)
        rich_texts   = sem_scorer._get_candidate_texts(unique_list)

        q_vecs = np.array(sem_scorer.embedding_function.embed_documents(batch_queries),  dtype=np.float32)
        d_vecs = np.array(sem_scorer.embedding_function.embed_documents(rich_texts),     dtype=np.float32)
        q_vecs /= (np.linalg.norm(q_vecs, axis=1, keepdims=True) + 1e-8)
        d_vecs /= (np.linalg.norm(d_vecs, axis=1, keepdims=True) + 1e-8)
        sim     = q_vecs @ d_vecs.T                       
        n2i     = {n: idx for idx, n in enumerate(unique_list)}

        for i, pool in enumerate(batch_pools):
            raw = {
                id2name[c]: sim[i, n2i[id2name[c]]]
                for c in pool
                if c in id2name and id2name[c] in n2i
            }
            sem_map[i] = _minmax(raw)

    results = []
    for i in range(B):
        pool  = batch_pools[i]
        s_seq = _minmax({id2name[c]: all_logits[i, c].item() for c in pool if c in id2name})

        s_gcn: Dict[str, float] = {}
        if all_gcn_logits is not None:
            raw_gcn = {}
            for c in pool:
                if c not in id2name: continue
                if c >= gcn_scorer.num_items or gcn_scorer.gcn_norm[c].norm().item() < 1e-6:
                    raw_gcn[id2name[c]] = -1.0  
                else:
                    raw_gcn[id2name[c]] = all_gcn_logits[i, c].item()
            s_gcn = _minmax(raw_gcn)

        results.append((s_seq, s_gcn, sem_map[i]))
    return results


def build_training_data_ce(
    loader,
    seq_scorer,
    gcn_scorer,
    sem_scorer,
    id2name:     Dict[int, str],
    cfg:         GatingConfig,
    dataset:     str   = 'amazon',
    balance_eps: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        X : (N, 7) context features
        Y : (N, 3) quality-proportional target gate distribution
    """
    query_prefix = DATASET_QUERY_PREFIX.get(dataset, 'User interested in: ')
    gcn_norm     = gcn_scorer.gcn_norm if gcn_scorer else None
    device       = seq_scorer.device
    X_list, Y_list = [], []
    skipped = 0

    pbar = tqdm(loader, desc='Building CE Training Data', unit='batch')
    for batch in pbar:
        seqs         = batch['seq'].to(device)
        lens         = batch['len_seq'].to(device)
        next_ids     = batch['next'].cpu().numpy()
        cans_t       = batch.get('cans')
        user_ids     = batch.get('id')

        batch_pools, batch_queries = [], []
        for i in range(len(next_ids)):
            gt_id = int(next_ids[i])
            cans  = cans_t[i].tolist() if cans_t is not None else []
            batch_pools.append(list(set(cans) | {gt_id}))

            seq_str = ' '.join(id2name[iid] for iid in seqs[i].tolist() if iid in id2name)
            batch_queries.append(query_prefix + seq_str)

        results = compute_batch_scores(
            seqs, lens, batch_pools, batch_queries, user_ids,
            seq_scorer, gcn_scorer, sem_scorer, id2name, device,
        )

        for i in range(len(next_ids)):
            gt_id   = int(next_ids[i])
            gt_name = id2name.get(gt_id)
            if not gt_name: continue

            s_seq, s_gcn, s_sem = results[i]

            ctx = extract_context_features(
                seq=seqs[i].tolist(), len_seq=int(lens[i]),
                seq_scores=s_seq, gcn_scores=s_gcn, sem_scores=s_sem,
                gcn_norm=gcn_norm, cfg=cfg,
            )

            q_arr = np.array([
                _ndcg_quality(gt_name, s_seq),
                _ndcg_quality(gt_name, s_gcn) if s_gcn else 0.0,
                _ndcg_quality(gt_name, s_sem) if s_sem else 0.0,
            ], dtype=np.float32)

            target_gate = _quality_proportional_target(q_arr, min_quality=balance_eps)
            if target_gate is None:
                skipped += 1
                continue

            X_list.append(ctx)
            Y_list.append(target_gate)

        pbar.set_postfix({'N': len(X_list), 'skipped': skipped})
    pbar.close()

    if skipped > 0:
        print(f'  ⚠️  Skipped {skipped:,} samples (no expert passed balance_eps={balance_eps})')
    return np.array(X_list, dtype=np.float32), np.array(Y_list, dtype=np.float32)


def train_gating_ce(
    X:         np.ndarray,
    Y_target:  np.ndarray,
    cfg:       GatingConfig,
    device:    torch.device,
    val_ratio: float = 0.1,
) -> GatingMLP:
    """
    Train GatingMLP with Cross-Entropy loss.

    Loss = −Σ target_i · log(pred_i)

    All data is loaded onto GPU at once (full-tensor mode) to avoid
    per-batch CPU→GPU overhead.
    """
    n     = len(X)
    n_val = max(1, int(n * val_ratio))
    idx   = np.random.permutation(n)
    v_idx, t_idx = idx[:n_val], idx[n_val:]
    n_tr  = len(t_idx)

    X_tr = torch.tensor(X[t_idx],        dtype=torch.float32, device=device)
    Y_tr = torch.tensor(Y_target[t_idx], dtype=torch.float32, device=device)
    X_vl = torch.tensor(X[v_idx],        dtype=torch.float32, device=device)
    Y_vl = torch.tensor(Y_target[v_idx], dtype=torch.float32, device=device)

    model     = GatingMLP(cfg).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=1e-5)

    B         = cfg.batch_size
    n_batches = math.ceil(n_tr / B)
    best_val, best_state = float('inf'), None

    for epoch in range(cfg.epochs):
        model.train()
        perm       = torch.randperm(n_tr, device=device)
        X_sh, Y_sh = X_tr[perm], Y_tr[perm]

        total_loss = 0.0
        for b in range(n_batches):
            x_b = X_sh[b * B : (b + 1) * B]
            y_b = Y_sh[b * B : (b + 1) * B]
            optimizer.zero_grad()
            gates = model(x_b)
            loss  = -(y_b * gates.log().clamp(min=-100)).sum(dim=-1).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            vl_gates = model(X_vl)
            val_loss = -(Y_vl * vl_gates.log().clamp(min=-100)).sum(dim=-1).mean().item()

        if val_loss < best_val:
            best_val   = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                avg_g = model(X_vl).cpu().numpy().mean(axis=0)
            print(f'  [CE] Epoch {epoch+1:03d}/{cfg.epochs} | '
                  f'train={total_loss / n_batches:.4f} | val={val_loss:.4f} | '
                  f'gates: seq={avg_g[0]:.3f} gcn={avg_g[1]:.3f} sem={avg_g[2]:.3f}')

    if best_state:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        X_all = torch.tensor(X, dtype=torch.float32, device=device)
        all_g = model(X_all).cpu().numpy()

    print(f'\n📊 Final avg gates (all data): '
          f'seq={all_g[:,0].mean():.3f}±{all_g[:,0].std():.3f} | '
          f'gcn={all_g[:,1].mean():.3f}±{all_g[:,1].std():.3f} | '
          f'sem={all_g[:,2].mean():.3f}±{all_g[:,2].std():.3f}')
    print(f'\n📊 Target gate distribution: '
          f'seq={Y_target[:,0].mean():.3f} | '
          f'gcn={Y_target[:,1].mean():.3f} | '
          f'sem={Y_target[:,2].mean():.3f}')
    print(f'\nBest val loss: {best_val:.4f}')
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',    required=True)
    parser.add_argument('--model_path',  required=True)
    parser.add_argument('--gcn_path',    default=None)
    parser.add_argument('--faiss_path',  default=None)
    parser.add_argument('--output_dir',  default='./saved_models/moe')
    parser.add_argument('--dataset',     default='amazon', choices=['amazon', 'yelp', 'goodreads', 'amazon_industrial', 'amazon_musical'])
    parser.add_argument('--epochs',      type=int,   default=50)
    parser.add_argument('--lr',          type=float, default=1e-3)
    parser.add_argument('--batch_size',  type=int,   default=256)
    parser.add_argument('--balance_eps', type=float, default=0.1,
                        help='Min NDCG threshold. Experts below threshold get weight=0. '
                             'Sample is skipped if all experts are below threshold.')
    parser.add_argument('--split',       default='val', choices=['train', 'val', 'test'])
    parser.add_argument('--hidden_size', type=int,   default=64)
    parser.add_argument('--embed_model', default='sentence-transformers/all-MiniLM-L6-v2')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[train_gating] Device={device} | dataset={args.dataset} | split={args.split}')
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load SASRec ──────────────────────────────────────────────────────────
    from utils.model import SASRec
    data_statis = pd.read_pickle(os.path.join(args.data_dir, 'data_statis.df'))
    seq_size, item_num = data_statis['seq_size'][0], data_statis['item_num'][0]
    ckpt      = torch.load(args.model_path, map_location=device, weights_only=False)
    hidden_sz = ckpt.get('hidden_size', args.hidden_size) if isinstance(ckpt, dict) else args.hidden_size
    sasrec    = SASRec(hidden_sz, item_num, seq_size, 0.1, device).to(device)
    sasrec.load_state_dict(ckpt.get('model_state_dict', ckpt))
    sasrec.eval()

    # ── id2name ──────────────────────────────────────────────────────────────
    id2rawid: Dict[int, str] = {}
    rawid_path = os.path.join(args.data_dir, 'id2rawid.txt')
    if os.path.exists(rawid_path):
        with open(rawid_path, encoding='utf-8') as f:
            for line in f:
                ll = line.strip().split('::')
                if len(ll) >= 2:
                    id2rawid[int(ll[0])] = ll[1].strip()

    id2name: Dict[int, str] = {}
    name2id: Dict[str, int] = {}
    with open(os.path.join(args.data_dir, 'id2name.txt'), encoding='utf-8') as f:
        for line in f:
            ll = line.strip().split('::', 1)
            if len(ll) == 2:
                cid       = int(ll[0])
                orig_name = ll[1].strip()
                raw_id    = id2rawid.get(cid, str(cid))
                unique_name      = f"{orig_name} [{raw_id}]"
                id2name[cid]     = unique_name
                name2id[unique_name] = cid

    shared = {
        'model': sasrec, 'id2name': id2name, 'name2id': name2id,
        'item_num': item_num, 'device': device, 'dataset': args.dataset,
    }
    seq_scorer = SeqScorer.from_shared(shared)

    # ── GCN ──────────────────────────────────────────────────────────────────
    gcn_scorer = None
    if args.gcn_path and os.path.exists(args.gcn_path):
        gcn_data = torch.load(args.gcn_path, map_location=device, weights_only=False)
        if isinstance(gcn_data, dict) and 'item_emb' in gcn_data:
            shared['gcn_embeddings'] = gcn_data['item_emb']
            shared['node_embs']      = gcn_data.get('node_embs', {})
            print(f'[GCN] Loaded remapped: item_emb={shared["gcn_embeddings"].shape}, '
                  f'node_embs={len(shared["node_embs"]):,} nodes')
        else:
            shared['gcn_embeddings'] = gcn_data
            shared['node_embs']      = {}
            print(f'[GCN] Loaded legacy Tensor: shape={shared["gcn_embeddings"].shape}')
        gcn_scorer = GCNScorer.from_shared(shared)

    # ── Semantic ─────────────────────────────────────────────────────────────
    sem_scorer = None
    if args.faiss_path and os.path.exists(args.faiss_path):
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_community.vectorstores import FAISS
        embed_fn = HuggingFaceEmbeddings(model_name=args.embed_model)
        vs       = FAISS.load_local(
            folder_path=args.faiss_path, embeddings=embed_fn,
            allow_dangerous_deserialization=True,
        )
        shared.update({'vector_store': vs, 'embedding_function': embed_fn})
        sem_scorer = SemanticScorer.from_shared(shared)
        print('[Semantic] FAISS loaded')

    # ── Dataset ──────────────────────────────────────────────────────────────
    class _Args:
        def __init__(self, d): self.data_dir = d

    dataset_obj = GeneralDataset(
        _Args(args.data_dir),
        stage={'train': 'train', 'val': 'valid', 'test': 'test'}[args.split],
    )
    loader = DataLoader(dataset_obj, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # ── GatingConfig ─────────────────────────────────────────────────────────
    cfg = GatingConfig(
        input_dim                = 7,
        hidden_dims              = [32, 16],
        dropout                  = 0.2,
        epochs                   = args.epochs,
        lr                       = args.lr,
        batch_size               = args.batch_size,
        gating_mode              = 'context',
        expert_quality_threshold = args.balance_eps,
    )

    # ── Build training data & Train ───────────────────────────────────────────
    print(f'\nBuilding CE training data (7-feature, balance_eps={args.balance_eps})...')
    X_raw, Y_target = build_training_data_ce(
        loader, seq_scorer, gcn_scorer, sem_scorer, id2name,
        cfg=cfg, dataset=args.dataset, balance_eps=args.balance_eps,
    )
    print(f'Training samples: {len(X_raw):,}')
    print_feature_report(X_raw)
    X_norm, mu, std = normalize_features(X_raw)

    print('\n🚀 Training GatingMLP (CE loss)...')
    trained_mlp = train_gating_ce(X_norm, Y_target, cfg, device)

    # ── Save ─────────────────────────────────────────────────────────────────
    out_path = os.path.join(args.output_dir, f'{args.dataset}_gating_model.pt')
    torch.save({
        'model_state_dict': trained_mlp.state_dict(),
        'cfg':              cfg,
        'norm_mean':        mu.tolist(),
        'norm_std':         std.tolist(),
        'gating_mode':      'context',
        'feature_version':  'v2.3',
        'feature_names':    FEATURE_LABELS,
        'loss':             'ce',
    }, out_path)
    print(f'\n✅ Saved → {out_path}')


if __name__ == '__main__':
    main()