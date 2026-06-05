import torch
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gcn_path',    required=True)
    parser.add_argument('--id2rawid',    required=True)
    parser.add_argument('--output_path', required=True)
    parser.add_argument('--emb_dim',     type=int, default=64)
    args = parser.parse_args()

    # ── 1. Load GCN dict: {original_id_str → tensor(64,)} ───────────────
    print(f"Loading GCN embeddings from {args.gcn_path}...")
    gcn_dict = torch.load(args.gcn_path, map_location='cpu')
    print(f"  Total nodes in GCN: {len(gcn_dict):,}")
    print(f"  Sample keys: {list(gcn_dict.keys())[:3]}")

    # ── 2. Load id2rawid: int_item_id → original_id (ASIN / MD5 / ...) ──
    print(f"\nLoading id2rawid from {args.id2rawid}...")
    int2raw = {}
    with open(args.id2rawid, encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('::', 1)
            if len(parts) == 2:
                int2raw[int(parts[0])] = parts[1].strip()

    print(f"  Total item mappings: {len(int2raw):,}")
    print(f"  Sample: {list(int2raw.items())[:3]}")

    # ── 3. Build item_emb tensor: index = int_item_id ────────────────────
    max_id  = max(int2raw.keys())
    emb_dim = args.emb_dim
    item_emb = torch.zeros(max_id + 1, emb_dim, dtype=torch.float32)

    matched   = 0
    unmatched = 0

    for int_id, raw_id in int2raw.items():
        if raw_id in gcn_dict:
            item_emb[int_id] = gcn_dict[raw_id].float()
            matched += 1
        else:
            unmatched += 1 

    print(f"\n  Matched:   {matched:,} / {len(int2raw):,}")
    print(f"  Unmatched: {unmatched:,} (zero vectors — cold items)")
    print(f"  Coverage:  {matched / len(int2raw) * 100:.1f}%")

    if matched == 0:
        raise ValueError(
            "0 items matched! id2rawid values không khớp với GCN keys.\n"
            f"  id2rawid sample: {list(int2raw.values())[:3]}\n"
            f"  GCN key sample:  {list(gcn_dict.keys())[:3]}"
        )

    node_embs = {str(k): v.float() for k, v in gcn_dict.items()}
    print(f"\n  node_embs: {len(node_embs):,} nodes (users + items)")

    output = {
        'item_emb':  item_emb,   
        'node_embs': node_embs,  
    }
    torch.save(output, args.output_path)
    print(f"\n✅ Saved to {args.output_path}")
    print(f"   item_emb shape: {item_emb.shape}")
    print(f"   node_embs keys: {len(node_embs):,}")

    print("\n--- Sanity Check ---")
    sample_ids = list(int2raw.keys())[:3]
    for iid in sample_ids:
        raw_id = int2raw[iid]
        if raw_id in gcn_dict:
            orig  = gcn_dict[raw_id].float()
            remap = item_emb[iid]
            ok    = torch.allclose(orig, remap)
            print(f"  int_id={iid} | raw_id={raw_id} | item_emb match={ok}")

    raw_keys  = set(int2raw.values())
    user_keys = [k for k in node_embs if k not in raw_keys]
    if user_keys:
        sample_user = user_keys[0]
        print(f"  Sample user node: '{sample_user}' → shape={node_embs[sample_user].shape} ✅")
    else:
        print(" Error user node ")


if __name__ == '__main__':
    main()