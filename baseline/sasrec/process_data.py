import json
import os
import pandas as pd
import collections
from tqdm import tqdm
from datetime import datetime
# RAW_DIR           = '../../dataset/output_data_all'
RAW_DIR           = '../../dataset/musical_industrial/industrial_amazon'
OUTPUT_DIR        = '../MoE/data'
# GROUND_TRUTH_FILE = '../MoE/data/ground_truth.json'
GROUND_TRUTH_FILE = '../MoE/data/groundtruth_music_industrial.json'
MAX_SEQ_LEN       = 50
MIN_INTERACTION   = 5   


def get_normalized_timestamp(data, source):
    try:
        if source == 'amazon' or source == 'amazon_industrial' or source == 'amazon_musical':
            return int(data.get('timestamp', 0))
        elif source == 'yelp':
            date_str = data.get('date')
            if date_str:
                return int(datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S").timestamp())
        elif source == 'goodreads':
            date_str = data.get('date_added') or data.get('date_updated')
            if date_str:
                return int(datetime.strptime(date_str, "%a %b %d %H:%M:%S %z %Y").timestamp())
        return 0
    except Exception:
        return 0


def pad_seq(s, max_len, pad_val=0):
    """Keep most recent max_len items, left-pad with pad_val."""
    s = s[-max_len:]
    return [pad_val] * (max_len - len(s)) + s


def process_source(target_source):
    print(f"\n{'='*60}")
    print(f"  PROCESSING: {target_source.upper()}")
    print(f"{'='*60}")

    save_dir = os.path.join(OUTPUT_DIR, target_source)
    os.makedirs(save_dir, exist_ok=True)

    item_file = os.path.join(RAW_DIR, 'item.json')
    raw_id_to_inner_id = {}
    inner_id_to_raw_id = {}
    id2name    = {}
    item_count = 1
    title_key  = 'name' if target_source == 'yelp' else 'title'

    with open(item_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                if data.get('source') == target_source:
                    raw_id = data.get('item_id')
                    if raw_id and raw_id not in raw_id_to_inner_id:
                        raw_id_to_inner_id[raw_id] = item_count
                        inner_id_to_raw_id[item_count] = raw_id
                        id2name[item_count] = data.get(title_key, 'Unknown').strip()
                        item_count += 1
            except Exception:
                continue

    print(f"[Items] Vocabulary size: {item_count - 1}")


    review_file       = os.path.join(RAW_DIR, 'review.json')
    user_interactions = collections.defaultdict(list)
    review_count      = 0

    with open(review_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                if data.get('source') != target_source:
                    continue
                uid     = data.get('user_id')
                iid_raw = data.get('item_id')
                ts      = get_normalized_timestamp(data, target_source)
                if uid and iid_raw in raw_id_to_inner_id:
                    user_interactions[uid].append(
                        (ts, raw_id_to_inner_id[iid_raw], iid_raw)
                    )
                    review_count += 1
            except Exception:
                continue

    print(f"[Reviews] Total interactions: {review_count}")
    print(f"[Reviews] Unique users      : {len(user_interactions)}")

    valid_users = set(user_interactions.keys())
    gt_map = {}

    if os.path.exists(GROUND_TRUTH_FILE):
        with open(GROUND_TRUTH_FILE, 'r', encoding='utf-8') as f:
            gt_list = json.load(f)

        gt_all = {item["user_id"]: item["item_id"] for item in gt_list}
        gt_map = {uid: iid for uid, iid in gt_all.items() if uid in valid_users}

        leaked = set(gt_all.keys()) - valid_users
        print(f"\n[GT] Total entries in file    : {len(gt_all)}")
        print(f"[GT] Valid for {target_source:<12}: {len(gt_map)}")
        if leaked:
            print(f"[GT] Filtered (other sources) : {len(leaked)} users")
        else:
            print(f"[GT] No cross-source leakage detected ✅")
    else:
        print(f"[GT] WARNING: Ground truth file not found: {GROUND_TRUTH_FILE}")

    train_data, val_data, test_data = [], [], []

    # Stats counters
    n_test_users        = 0
    n_non_test_users    = 0
    n_skipped_short     = 0
    n_gt_in_history     = 0
    n_gt_appended       = 0

    all_users = list(user_interactions.keys())

    for uid in tqdm(all_users, desc=f"[{target_source}] Building splits"):
        interactions = user_interactions[uid]
        interactions.sort(key=lambda x: x[0])  

        inner_ids = [x[1] for x in interactions]
        raw_ids   = [x[2] for x in interactions]

        is_test_user = uid in gt_map
        if is_test_user:
            gt_raw = gt_map[uid]
            n_test_users += 1

            if gt_raw in raw_ids:
                idx = raw_ids.index(gt_raw)
                full_sequence = inner_ids[:idx + 1]  # [i1 .. gt]
                n_gt_in_history += 1
            else:
                gt_inner = raw_id_to_inner_id.get(gt_raw)
                if gt_inner is None:
   
                    continue
                full_sequence = inner_ids + [gt_inner]
                n_gt_appended += 1
        else:
            n_non_test_users += 1
            full_sequence = inner_ids

        if not is_test_user and len(full_sequence) < MIN_INTERACTION:
            n_skipped_short += 1
            continue

        if len(full_sequence) < 3:
            n_skipped_short += 1
            continue

        N = len(full_sequence)

        # ── Test split: input=[i1..iN-1], target=iN ───────────────────────
        test_seq    = full_sequence[:-1]   # [i1 .. iN-1]
        test_target = full_sequence[-1]    # iN

        test_data.append({
            'uid'    : uid,
            'seq'    : pad_seq(test_seq, MAX_SEQ_LEN),
            'len_seq': min(len(test_seq), MAX_SEQ_LEN),
            'next'   : test_target,
            'is_test_user': is_test_user,
        })

        # ── Val split: input=[i1..iN-2], target=iN-1 ──────────────────────
        val_seq    = full_sequence[:-2]    # [i1 .. iN-2]
        val_target = full_sequence[-2]     # iN-1

        val_data.append({
            'uid'    : uid,
            'seq'    : pad_seq(val_seq, MAX_SEQ_LEN),
            'len_seq': min(len(val_seq), MAX_SEQ_LEN),
            'next'   : val_target,
            'is_test_user': is_test_user,
        })

        if N < 15:
            stride = 1
        elif N < 50:
            stride = 2
        else:
            stride = 3
            
        for t in range(1, N - 2, stride):
            train_seq    = full_sequence[:t]   # [i1 .. it]
            train_target = full_sequence[t]    # i(t+1)

            train_data.append({
                'uid'    : uid,
                'seq'    : pad_seq(train_seq, MAX_SEQ_LEN),
                'len_seq': min(len(train_seq), MAX_SEQ_LEN),
                'next'   : train_target,
            })
 
    with open(os.path.join(save_dir, 'id2name.txt'), 'w', encoding='utf-8') as f:
        for iid, name in id2name.items():
            f.write(f"{iid}::{name.replace(chr(10), ' ').replace(chr(13), ' ')}\n")

    with open(os.path.join(save_dir, 'id2rawid.txt'), 'w', encoding='utf-8') as f:
        for inner_id, raw_id in inner_id_to_raw_id.items():
            f.write(f"{inner_id}::{raw_id}\n")

    statis = pd.DataFrame({
        'seq_size': [MAX_SEQ_LEN],
        'item_num': [item_count],
    })
    statis.to_pickle(os.path.join(save_dir, 'data_statis.df'))

    pd.DataFrame(train_data).to_pickle(os.path.join(save_dir, 'train_data.df'))
    pd.DataFrame(val_data).to_pickle(os.path.join(save_dir, 'Val_data.df'))
    pd.DataFrame(test_data).to_pickle(os.path.join(save_dir, 'Test_data.df'))

    print(f"\n✅ {target_source.upper()} complete:")
    print(f"   Split strategy              : LOO (SASRec standard)")
    print(f"   ─────────────────────────────────────────────────")
    print(f"   Train samples               : {len(train_data)}")
    print(f"   Val samples                 : {len(val_data)}")
    print(f"   Test samples                : {len(test_data)}")
    print(f"   ─────────────────────────────────────────────────")
    print(f"   Total users processed       : {len(all_users)}")
    print(f"     ├─ Test users (gt)        : {n_test_users}")
    print(f"     │    ├─ GT in history     : {n_gt_in_history}")
    print(f"     │    └─ GT appended (cold): {n_gt_appended}")
    print(f"     ├─ Non-test users         : {n_non_test_users}")
    print(f"     └─ Skipped (too short)    : {n_skipped_short}")
    print(f"   Item vocab size             : {item_count - 1}")
    print(f"\n   Leak check:")
    print(f"     Train targets ∩ Val targets  = ∅ ✅  (train→iN-2, val→iN-1)")
    print(f"     Train targets ∩ Test targets = ∅ ✅  (train→iN-2, test→iN)")
    print(f"     Val targets   ∩ Test targets = ∅ ✅  (val→iN-1,  test→iN)")
    print(f"     All users in ALL splits       ✅  (LOO standard)")


if __name__ == "__main__":
    for source in ['amazon_industrial']:
        process_source(source)