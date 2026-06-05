import os
import json
from tqdm import tqdm


def build_candidate_order(candidate_dir):
    order_map = {}
    if not candidate_dir or not os.path.exists(candidate_dir):
        return order_map

    candidate_files = [f for f in os.listdir(candidate_dir)
                       if f.startswith('task_') and f.endswith('.json')]
    # Sort by file number: task_0, task_1, ..., task_100
    candidate_files.sort(key=lambda f: int(f.replace('task_', '').replace('.json', '')))

    for idx, file_name in enumerate(candidate_files):
        try:
            with open(os.path.join(candidate_dir, file_name), 'r', encoding='utf-8') as f:
                data = json.load(f)
                entry = data[0] if isinstance(data, list) else data
                uid = str(entry.get('user_id'))
                order_map[uid] = idx
        except:
            continue

    print(f"[CandidateOrder] Built order map for {len(order_map)} users "
          f"(task_0 → task_{len(order_map)-1})")
    return order_map


def load_candidate_map(candidate_dir):
    candidate_map = {}
    if not candidate_dir or not os.path.exists(candidate_dir):
        return candidate_map
    
    candidate_files = [f for f in os.listdir(candidate_dir) if f.startswith('task_') and f.endswith('.json')]
    for file_name in tqdm(candidate_files, desc="Loading Candidates"):
        try:
            with open(os.path.join(candidate_dir, file_name), 'r', encoding='utf-8') as f:
                data = json.load(f)
                entry = data[0] if isinstance(data, list) else data
                candidate_map[str(entry.get('user_id'))] = entry.get('candidate_list', [])
        except: continue
    return candidate_map

def load_item_name_map(mapping_file):
    name_map = {}
    if not mapping_file or not os.path.exists(mapping_file):
        return name_map
    with open(mapping_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line)
                name = item.get('title') or item.get('name') or "Unknown"
                asin = item.get('asin') or item.get('item_id') or "Unknown"
                unique_name = f"{name.strip()} [{asin.strip()}]"
                name_map[str(item.get('item_id')).strip()] = unique_name
            except: continue
    return name_map
def prepare_merge_data(new_input_list, data_map, candidate_map, item_id_to_name_map, user_agent_sasrec, args):
    merge_data_list = []
    skipped = 0
    padding_id = 0
    rawid2id = {}
    if hasattr(user_agent_sasrec, 'id2rawid') and user_agent_sasrec.id2rawid:
        rawid2id = {v: k for k, v in user_agent_sasrec.id2rawid.items()}
    
    total_missing = 0
    total_invalid = 0
    missing_examples = []  

    for entry in tqdm(new_input_list, desc="Merging Data"):
        user_id = str(entry.get('user_id'))
        gt_item_id = str(entry.get('item_id', '')).strip()
        
        gt_inner_id = rawid2id.get(gt_item_id)
        name_from_map = item_id_to_name_map.get(gt_item_id)
        name_from_model = user_agent_sasrec.id2name.get(gt_inner_id) if gt_inner_id is not None else \
                          user_agent_sasrec.id2name.get(int(gt_item_id) if gt_item_id.isdigit() else -1)
        
        if not name_from_map and not name_from_model:
            print(f"\n[CRITICAL] Ground Truth Item {gt_item_id} (User: {user_id}) not found in item.json or id2name.txt!")
        
        gt_item_name = name_from_map or name_from_model or gt_item_id

        if user_id not in data_map:
            data = {'id': user_id, 'uid': user_id, 'seq': [padding_id] * user_agent_sasrec.seq_size, 'seq_str': "Empty History", 'len_seq': 0, 'seq_unpad': []}
        else:
            data = data_map[user_id].copy()

        data['correct_answer'] = gt_item_name
        
        if user_id in candidate_map:
            new_ids, new_names = [], []
            for rid in candidate_map[user_id]:
                rid_str = str(rid).strip()
                
                # Step 1: look up ASIN → inner_id via rawid2id
                iid = rawid2id.get(rid_str)

                if iid is not None:
                    name = user_agent_sasrec.id2name.get(iid, rid_str)
                else:
                    # Step 2: fallback — try name map then name2id (for numeric inner_id)
                    name = item_id_to_name_map.get(rid_str) or user_agent_sasrec.id2name.get(int(rid_str) if rid_str.isdigit() else -1) or rid_str
                    iid = user_agent_sasrec.name2id.get(name)

                # Item not in SASRec model (cold-start item)
                if iid is None:
                    total_missing += 1
                    if len(missing_examples) < 5:
                        missing_examples.append(f"  '{name}' (ID: {rid_str})")
                    continue
                
                if iid >= user_agent_sasrec.item_num:
                    total_invalid += 1
                    continue

                new_names.append(name)
                new_ids.append(iid)
            
            if len(new_ids) == 0:
                print(f"[WARNING] User {user_id} has no valid candidates after filtering!")
                skipped += 1
                continue

            data['cans'] = new_ids
            data['cans_str'] = args.sep.join(new_names)
            data['len_cans'] = len(new_ids)
            data['cans_name'] = new_names
        else:
            skipped += 1
            continue

        try:
            data['prior_answer'] = user_agent_sasrec.model_generate(data['seq'], data['len_seq'], data['cans'])
            merge_data_list.append(data)
        except Exception as e:
            print(f"\n[ERROR] model_generate failed for User {user_id}: {e}")
            print(f"  Cans: {data.get('cans')}, Item_num: {user_agent_sasrec.item_num}")
            skipped += 1
    
    if total_missing > 0:
        print(f"\n[MISSING ITEMS SUMMARY] {total_missing} candidate items not found in SASRec model (cold-start items filtered during preprocessing).")
        print(f"  → Cause: Candidate list contains ASINs not in id2rawid.txt/id2name.txt")
        print(f"  → These items were skipped (does not affect results).")
        if missing_examples:
            print(f"  → Examples ({len(missing_examples)} samples):")
            for ex in missing_examples:
                print(ex)
    if total_invalid > 0:
        print(f"[INVALID IDs SUMMARY] {total_invalid} items with inner_id >= item_num were discarded.")
            
    return merge_data_list, skipped