import argparse
import os
import sys
import json
import math
import random
import time
import multiprocessing
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import pandas as pd
current_dir  = os.path.dirname(os.path.abspath(__file__))  # MoE/
project_root = os.path.dirname(current_dir)               # moe_early_fusion/

for _p in [current_dir, project_root]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

baseline_dir = os.path.join(project_root, 'baseline')
if baseline_dir not in sys.path:
    sys.path.insert(0, baseline_dir)

from utils.data_processor import (
    load_candidate_map, load_item_name_map,
    prepare_merge_data, build_candidate_order,
)
from utils.helper_function import save_final_metrics, split_user_response,append_jsonl
from dataset.general_dataset import GeneralDataset
from utils.agent import UserModelAgent
from core.moe_rec_agent import MoERecAgent
from websocietysimulator.tools import CacheInteractionTool


def get_args():
    parser = argparse.ArgumentParser(description="MoE Early Fusion — standalone runner")
    parser.add_argument('--data_dir',          type=str, required=True)
    parser.add_argument('--model_path',        type=str, required=True)
    parser.add_argument('--input_json_file',   type=str, required=True)
    parser.add_argument('--candidate_dir',     type=str, default=None)
    parser.add_argument('--item_mapping_file', type=str, default=None)
    parser.add_argument('--raw_data_dir',      type=str, default=None)
    parser.add_argument('--stage',             type=str, default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--dataset',           type=str, default='amazon', choices=['amazon', 'yelp', 'goodreads', 'amazon_musical', 'amazon_industrial'])
    parser.add_argument('--cans_num',  type=int, default=20)
    parser.add_argument('--max_epoch', type=int, default=3)
    parser.add_argument('--faiss_db_path', type=str, default=None)
    parser.add_argument('--gcn_path',      type=str, default=None)
    parser.add_argument('--embed_model_name', type=str, default='sentence-transformers/all-MiniLM-L6-v2')
    parser.add_argument('--gating_model_path', type=str, default=None)
    parser.add_argument('--reranker_mode', type=str, default='embed_only', choices=['embed_only', 'llm', 'hybrid'])
    parser.add_argument('--reranker_top_llm', type=int, default=15)
    parser.add_argument('--use_reranker',   type=lambda x: x.lower() in ('true','1','yes'), default=True, help='Enable/disable Reranker LLM (default: True)')
    parser.add_argument('--use_user_agent', type=lambda x: x.lower() in ('true','1','yes'), default=True, help='Enable/disable User Agent LLM (default: True)')
    parser.add_argument('--rerank_only', action='store_true')
    # ── Ablation hyperparameters ──────────────────────────────────────────────
    parser.add_argument('--alpha', type=float, default=None,
                        help='Fusion weight α: score = α*MoE + (1-α)*Reranker. '
                             'Overrides config.py ScoreConfig.alpha (default: 0.5). '
                             'Ablation sweep: {0.1, 0.2, ..., 0.9}')
    parser.add_argument('--top_m', type=int, default=None,
                        help='Pool size M: number of candidates after MoE Fusion before Reranker. '
                             'Overrides config.py RetrievalConfig.top_M (default: 20). '
                             'Ablation sweep: {10, 15, 20, 25, 30}')
    parser.add_argument('--model',       type=str, default='qwen-small')
    parser.add_argument('--api_key',     type=str, default=None)
    parser.add_argument('--base_url',    type=str, default=None,
                        help='LLM API base URL. None/empty = OpenAI default (api.openai.com). Set to local vLLM URL when using local model.')

    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--mp',             type=int, default=4)
    parser.add_argument('--seed',           type=int, default=333)
    parser.add_argument('--max_samples',    type=int, default=-1)
    parser.add_argument('--max_retry_num',  type=int, default=5)
    parser.add_argument('--hidden_size',    type=int, default=64)
    parser.add_argument('--dropout',        type=float, default=0.1)
    parser.add_argument('--sep',            type=str, default=', ')
    parser.add_argument('--output_file',   type=str, default='./output/moe_results.jsonl')
    parser.add_argument('--result_file',   type=str, default='./output/moe_evaluation_summary.json')
    parser.add_argument('--save_info',     action='store_true')
    parser.add_argument('--save_rec_dir',  type=str, default='./output/moe_rec_logs')
    parser.add_argument('--save_user_dir', type=str, default='./output/moe_user_logs')
    return parser.parse_args()

def recommend_moe(data: dict, args) -> tuple:
    
    user_id    = data.get('id')
    llm_client = None
    
    # Initialize LLM only when required (Reranker or User Agent)
    need_llm = getattr(args, 'use_reranker', True) or getattr(args, 'use_user_agent', True)
    if need_llm:
        try:
            from langchain_openai import ChatOpenAI
            key = args.api_key if args.api_key and args.api_key.lower() not in ('none', 'empty') else "EMPTY"
            base_url = getattr(args, 'base_url', None)

            llm_kwargs = dict(
                model           = args.model,
                openai_api_key  = key,
                temperature     = args.temperature,
                max_retries     = 5,
                request_timeout = 120,
                max_tokens      = 800,
            )
            # Only set base_url when a real value is provided (local vLLM).
            # When empty/None, langchain uses https://api.openai.com/v1 by default.
            if base_url and base_url.strip():
                llm_kwargs['openai_api_base'] = base_url.strip()

            is_openai = not (base_url and base_url.strip())
            backend   = f"OpenAI API ({args.model})" if is_openai else f"local vLLM @ {base_url}"
            print(f"[MoE][User {user_id}] LLM backend: {backend}")

            llm_client = ChatOpenAI(**llm_kwargs)
        except ImportError:
            print(f"[MoE] Warning: could not import langchain_openai for User {user_id}")
        except Exception as e:
            print(f"[MoE] LLM init error for User {user_id}: {e}")
    else:
        print(f"[MoE][User {user_id}] Skipping LLM init (use_reranker={getattr(args, 'use_reranker', True)}, use_user_agent={getattr(args, 'use_user_agent', True)})")

    rec_agent  = MoERecAgent(args, llm=llm_client)
    shared     = rec_agent.get_shared_sasrec()
    
    # User Agent: only initialize when enabled
    use_user_agent = getattr(args, 'use_user_agent', True)
    user_agent = UserModelAgent(args, shared_sasrec=shared) if use_user_agent else None

    flag          = False
    epoch         = 1
    rec_item_list = []
    new_data_list = []
    hit_at_n      = {1: False, 3: False, 5: False, 'rank': None}

    rejected_items: list = []
    _invalid_reasons = {'', 'Could not parse user response.', 'Fallback: MoE pipeline unavailable.', 'Fallback: recommendation agent unavailable.'}

    # --- DEBUG TRACKING VARIABLES ---
    gt_rank_history        = {}
    ndcg_history           = {}   # NDCG per epoch, used for reranker impact logging
    drop_logs              = []
    improve_logs           = []
    reranker_impact_logs   = []   # [RERANKER BOOST/DROP] after user rejection

    ndcg_v1        = 0.0
    ndcg_final     = 0.0
    moe_only_ndcg  = 0.0
    hit_v1         = {1: False, 3: False, 5: False}
    gate_records   = []
    score_records  = []
    # --------------------------------

    # If User Agent is disabled, run only one round (no feedback loop)
    effective_max_epoch = args.max_epoch if use_user_agent else 1
    
    while not flag and epoch <= effective_max_epoch:
        prefix = f"[MoE][User {user_id}][Round {epoch}]"

        max_retries      = 3
        rec_reason       = None
        current_rec_list = []
        debug_info       = {}

        for attempt in range(max_retries):
            try:
                rec_reason, current_rec_list, debug_info = rec_agent.act(
                    data           = data,
                    epoch          = epoch,
                    rejected_items = rejected_items,
                )
            except Exception as e:
                print(f"{prefix} act() error: {e}, retry {attempt+1}/{max_retries}")
                time.sleep(1)
                continue

            if rec_reason and current_rec_list:
                rec_item_list = current_rec_list
                break
        else:
            rec_item_list = data.get('cans_name', [])[:5]
            rec_reason    = "Fallback: MoE pipeline unavailable."
            debug_info    = {'error': 'Fallback trigger - All retries failed'}

        gt_name = data.get('correct_answer', '').strip()
        gt_name_lower       = gt_name.lower()
        current_top_n_lower = [item.lower().strip() for item in rec_item_list]
        
        if gt_name_lower in current_top_n_lower:
            current_rank = current_top_n_lower.index(gt_name_lower) + 1
        else:
            current_rank = 999 
            
        gt_rank_history[epoch] = current_rank
        current_ndcg = 1.0 / math.log2(current_rank + 1) if current_rank <= 5 else 0.0
        ndcg_history[epoch] = current_ndcg

        if epoch == 1:
            ndcg_v1 = current_ndcg
            if current_rank <= 1: hit_v1[1] = True
            if current_rank <= 3: hit_v1[3] = True
            if current_rank <= 5: hit_v1[5] = True

            # --- MOE ONLY NDCG (before Reranker) ---
            c_m_top_k = debug_info.get('c_m_top_k_before_rerank', [])
            c_m_top_k_lower = [item.lower().strip() for item in c_m_top_k]
            if gt_name_lower in c_m_top_k_lower:
                moe_rank = c_m_top_k_lower.index(gt_name_lower) + 1
                if moe_rank <= 5:
                    moe_only_ndcg = 1.0 / math.log2(moe_rank + 1)

        # --- GATES & SCORES TRACKING ---
        gates = debug_info.get('avg_gates', {})
        if gates:
            gate_records.append([gates.get('seq', 0), gates.get('gcn', 0), gates.get('sem', 0)])
        scores = debug_info.get('scores_breakdown', {})
        for it_name, s_vals in scores.items():
            score_records.append([s_vals.get('s0_moe_rank', s_vals.get('s0_moe', 0.0)), s_vals.get('s_rerank', 0.0)])
        # ----------------------------

        if epoch >= 2:
            prev_rank  = gt_rank_history.get(epoch - 1, 999)
            prev_ndcg  = ndcg_history.get(epoch - 1, 0.0)
            prev_reason = "N/A"
            if len(rec_agent.info_list) > 0:
                prev_reason = rec_agent.info_list[-1].get('user_reason', 'N/A').strip()

            str_prev_rank = f"Rank {prev_rank}" if prev_rank != 999 else "Out of List"
            str_curr_rank = f"Rank {current_rank}" if current_rank != 999 else "Out of List"
            short_reason  = (prev_reason[:120] + '...') if len(prev_reason) > 120 else prev_reason

            if current_rank > prev_rank:
                log_str = f"[WARNING] User {user_id} | GT: '{gt_name}' | Rank Drop: Round {epoch-1} ({str_prev_rank}) -> Round {epoch} ({str_curr_rank}) | Prev rejection: '{prev_reason}'"
                drop_logs.append(log_str)
            elif current_rank < prev_rank:
                log_str = f"[SUCCESS] User {user_id} | GT: '{gt_name}' | Rank Improve: Round {epoch-1} ({str_prev_rank}) -> Round {epoch} ({str_curr_rank}) | Prev rejection: '{prev_reason}'"
                improve_logs.append(log_str)
                print(log_str)

            # Reranker impact log (after user rejection)
            ndcg_delta = current_ndcg - prev_ndcg
            if ndcg_delta > 1e-6:
                ri_log = (
                    f"[RERANKER BOOST] User: {user_id} | GT: '{gt_name}' | "
                    f"Rejection: '{short_reason}' | "
                    f"NDCG: {prev_ndcg:.4f} -> {current_ndcg:.4f} (+{ndcg_delta:.4f}) | "
                    f"Round: {epoch-1}->{epoch}"
                )
                reranker_impact_logs.append(ri_log)
                print(ri_log)
            elif ndcg_delta < -1e-6:
                ri_log = (
                    f"[RERANKER DROP]  User: {user_id} | GT: '{gt_name}' | "
                    f"Rejection: '{short_reason}' | "
                    f"NDCG: {prev_ndcg:.4f} -> {current_ndcg:.4f} ({ndcg_delta:.4f}) | "
                    f"Round: {epoch-1}->{epoch}"
                )
                reranker_impact_logs.append(ri_log)
            # -------------------------------------------------

        max_user_retries    = 3
        user_agent_response = None
        user_reason         = None

        if use_user_agent and user_agent is not None:
            for attempt in range(max_user_retries):
                cache = rec_agent.sem_scorer._docstore_cache if hasattr(rec_agent, 'sem_scorer') else None
                user_agent_response = user_agent.act(data, rec_reason, rec_item_list, docstore_cache=cache)
                user_reason, flag   = split_user_response(user_agent_response)
                if user_reason is not None and flag is not None:
                    break
                time.sleep(1)
            else:
                user_reason = "Could not parse user response."
                flag        = False
        else:
            user_agent_response = "User Agent disabled — auto-accepting."
            user_reason = None
            flag = True

        user_reason_clean = (user_reason or '').strip()
        
        rec_res = (f"Reason: {rec_reason}\nItems: {', '.join(current_rec_list[:5])}")
        
        new_data_list.append({
            'id':              str(user_id),
            'epoch':           epoch,
            'gt_item':         gt_name,
            'rec_res':         rec_res,
            'user_res':        user_agent_response,
            'rec_items':       rec_item_list,
            'penalized_items': list(rejected_items),
            'flag':            flag,
            'pipeline':        'moe',
            'debug_rerank':    debug_info,
        })

        if flag or epoch == args.max_epoch:
            ndcg_final = current_ndcg
            if gt_name_lower in current_top_n_lower:
                rank             = current_top_n_lower.index(gt_name_lower) + 1
                hit_at_n['rank'] = rank
                if rank <= 1: hit_at_n[1] = True
                if rank <= 3: hit_at_n[3] = True
                if rank <= 5: hit_at_n[5] = True
            if flag: break

        if user_reason_clean and user_reason_clean not in _invalid_reasons:
            memory_info = {
                'epoch':         epoch,
                'rec_reason':    rec_reason,
                'rec_item_list': rec_item_list,
                'user_reason':   user_reason,
            }
            rec_agent.update_memory(memory_info)
            if use_user_agent and user_agent is not None:
                user_agent.update_memory(memory_info)

        epoch += 1

    return new_data_list, hit_at_n, args, drop_logs, improve_logs, ndcg_v1, ndcg_final, gate_records, score_records, hit_v1, moe_only_ndcg, reranker_impact_logs


def error_handler(e):
    import traceback
    traceback.print_exc()

def make_counters(manager):
    return {
        'finish_num': manager.Value('i', 0), 'correct_hit1': manager.Value('i', 0),
        'correct_hit3': manager.Value('i', 0), 'correct_hit5': manager.Value('i', 0),
        'total_hit1_v1': manager.Value('i', 0), 'total_hit3_v1': manager.Value('i', 0), 'total_hit5_v1': manager.Value('i', 0),
        'total_ndcg5': manager.Value('d', 0.0), 'total': manager.Value('i', 0),
        'total_feedback_triggered': manager.Value('i', 0), 
        'total_rank_drops': manager.Value('i', 0),         
        'total_rank_improves': manager.Value('i', 0),
        'total_ndcg_v1': manager.Value('d', 0.0),
        'total_ndcg_final': manager.Value('d', 0.0),
        'total_moe_only_ndcg': manager.Value('d', 0.0),
        'moe_better_count': manager.Value('i', 0),
        'reranker_better_count': manager.Value('i', 0),
        'equal_count': manager.Value('i', 0),
        'gate_vals': manager.list(),
        'score_vals': manager.list(),
        'lock': manager.Lock(),
    }

def setcallback_safe(result, counters, args):
    data_list, hit_at_n, _args, drop_logs, improve_logs, ndcg_v1, ndcg_final, gate_recs, score_recs, hit_v1, moe_only_ndcg, reranker_impact_logs = result
    for step in data_list: append_jsonl(args.output_file, step)

    output_dir = os.path.dirname(args.output_file) or '.'
    
    with counters['lock']:
        # 1. NDCG Comparison Log
        ndcg_log_path = os.path.join(output_dir, 'ndcg_comparison_log.txt')
        with open(ndcg_log_path, 'a', encoding='utf-8') as f:
            boost = ndcg_final - ndcg_v1
            f.write(f"User: {data_list[0]['id']} | V1 NDCG: {ndcg_v1:.4f} | Final NDCG: {ndcg_final:.4f} | Boost: {boost:+.4f}\n")

        # 1.5. MoE vs Reranker NDCG Comparison Log
        moe_vs_rerank_log_path = os.path.join(output_dir, 'moe_vs_reranker_ndcg_log.txt')
        with open(moe_vs_rerank_log_path, 'a', encoding='utf-8') as f:
            boost_by_reranker = ndcg_v1 - moe_only_ndcg
            f.write(f"User: {data_list[0]['id']} | MoE NDCG: {moe_only_ndcg:.4f} | Reranker NDCG: {ndcg_v1:.4f} | Reranker Boost: {boost_by_reranker:+.4f}\n")

        # 2. Rank Drops Log
        if drop_logs:
            drop_log_path = os.path.join(output_dir, 'feedback_rank_drop_log.txt')
            with open(drop_log_path, 'a', encoding='utf-8') as f:
                for log in drop_logs: f.write(log + '\n')
                    
        # 3. Rank Improves Log
        if improve_logs:
            improve_log_path = os.path.join(output_dir, 'feedback_rank_improve_log.txt')
            with open(improve_log_path, 'a', encoding='utf-8') as f:
                for log in improve_logs: f.write(log + '\n')

        # 4. Reranker Feedback Impact Log (BOOST / DROP after user rejection)
        if reranker_impact_logs:
            impact_log_path = os.path.join(output_dir, 'reranker_feedback_impact_log.txt')
            with open(impact_log_path, 'a', encoding='utf-8') as f:
                for log in reranker_impact_logs: f.write(log + '\n')

        # Update counters
        counters['finish_num'].value += 1
        if len(data_list) > 1: 
            counters['total_feedback_triggered'].value += 1
            
        counters['total_rank_drops'].value += len(drop_logs)
        counters['total_rank_improves'].value += len(improve_logs)
        
        counters['total_ndcg_v1'].value += ndcg_v1
        counters['total_ndcg_final'].value += ndcg_final
        counters['total_moe_only_ndcg'].value += moe_only_ndcg
        if ndcg_v1 > moe_only_ndcg:
            counters['reranker_better_count'].value += 1
        elif moe_only_ndcg > ndcg_v1:
            counters['moe_better_count'].value += 1
        else:
            counters['equal_count'].value += 1
        
        if hit_v1.get(1): counters['total_hit1_v1'].value += 1
        if hit_v1.get(3): counters['total_hit3_v1'].value += 1
        if hit_v1.get(5): counters['total_hit5_v1'].value += 1
        counters['gate_vals'].extend(gate_recs)
        counters['score_vals'].extend(score_recs)
        
        if hit_at_n.get(1): counters['correct_hit1'].value += 1
        if hit_at_n.get(3): counters['correct_hit3'].value += 1
        if hit_at_n.get(5): counters['correct_hit5'].value += 1
        rank = hit_at_n.get('rank')
        if rank is not None and rank <= 5:
            counters['total_ndcg5'].value += 1.0 / math.log2(rank + 1)
        
        fn, tot = counters['finish_num'].value, counters['total'].value
        h1, h3, h5 = counters['correct_hit1'].value, counters['correct_hit3'].value, counters['correct_hit5'].value
        ndcg = counters['total_ndcg5'].value

    print(f"[MoE][{fn}/{tot}] Hit@1: {h1/fn*100:.2f}% | Hit@3: {h3/fn*100:.2f}% | Hit@5: {h5/fn*100:.2f}% | NDCG@5: {ndcg/fn:.4f}", flush=True)


def main(args):
    args.use_moe  = True
    args.use_arag = False

    # Override config from CLI ablation args
    from config import DEFAULT_CONFIG
    if args.alpha is not None:
        alpha_val = float(np.clip(args.alpha, 0.0, 1.0))
        DEFAULT_CONFIG.scoring.alpha = alpha_val
        for k in DEFAULT_CONFIG.scoring.dataset_alpha:
            DEFAULT_CONFIG.scoring.dataset_alpha[k] = alpha_val
        print(f"[Ablation] α overridden → {alpha_val:.2f}")
    if args.top_m is not None:
        DEFAULT_CONFIG.retrieval.top_M = int(args.top_m)
        print(f"[Ablation] top_M overridden → {int(args.top_m)}")
    
    dataset  = GeneralDataset(args, stage=args.stage)
    data_map = {str(d['id']): d for d in dataset}
    with open(args.input_json_file, 'r', encoding='utf-8') as f: new_input_list = json.load(f)
    candidate_map = load_candidate_map(args.candidate_dir)
    item_name_map = load_item_name_map(args.item_mapping_file)

    import argparse as _ap
    temp_args = _ap.Namespace(**vars(args))
    temp_args.model = 'sasrec_inference'

    MoERecAgent._init_shared_resources(args)
    
    shared_sasrec_info = {
        'model': MoERecAgent._shared_sasrec_model, 
        'id2name': MoERecAgent._shared_id2name, 
        'name2id': MoERecAgent._shared_name2id, 
        'id2rawid': MoERecAgent._shared_id2rawid, 
        'seq_size': MoERecAgent._shared_seq_size, 
        'item_num': MoERecAgent._shared_item_num, 
        'device': MoERecAgent._shared_device
    }

    sasrec_tool = UserModelAgent(temp_args, mode='prior_rec', shared_sasrec=shared_sasrec_info)

    merge_data_list, skipped = prepare_merge_data(new_input_list, data_map, candidate_map, item_name_map, sasrec_tool, args)

    if not args.raw_data_dir: args.raw_data_dir = os.path.join(project_root, 'dataset', 'output_data_all')
    args.raw_data_dir = os.path.abspath(args.raw_data_dir)
    interaction_tool = CacheInteractionTool(data_dir=args.raw_data_dir)

    id2rawid = {}
    rawid_path = os.path.join(args.data_dir, 'id2rawid.txt')
    if os.path.exists(rawid_path):
        with open(rawid_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('::')
                if len(parts) >= 2: id2rawid[int(parts[0])] = parts[1].strip()

    for d in merge_data_list:
        d['interaction_tool'] = interaction_tool
        d['id2rawid']         = id2rawid

    candidate_order = build_candidate_order(args.candidate_dir)
    if candidate_order: merge_data_list.sort(key=lambda d: candidate_order.get(str(d['id']), float('inf')))
    if args.max_samples > 0: merge_data_list = merge_data_list[:args.max_samples]
    # if args.max_samples > 0: merge_data_list = merge_data_list[500:1000]

    os.makedirs(os.path.dirname(args.output_file) or '.', exist_ok=True)
    output_dir = os.path.dirname(args.output_file) or '.'
    
    # Reset log files
    for f_name in [
        'feedback_rank_drop_log.txt', 'feedback_rank_improve_log.txt',
        'ndcg_comparison_log.txt', 'moe_vs_reranker_ndcg_log.txt',
        'reranker_feedback_impact_log.txt',         # Reranker boost/drop after rejection
    ]:
        with open(os.path.join(output_dir, f_name), 'w', encoding='utf-8') as f:
            f.write(f"=== {f_name.upper()} ===\n")

    total = len(merge_data_list)
    effective_workers = max(1, args.mp)

    with multiprocessing.Manager() as manager:
        counters = make_counters(manager)
        counters['total'].value = total
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            futures = {executor.submit(recommend_moe, data, args): data for data in merge_data_list}
            for future in tqdm(as_completed(futures), total=total, desc="Processing (MoE)"):
                try: setcallback_safe(future.result(), counters, args)
                except Exception as e: error_handler(e)

        final_hit1, final_hit3, final_hit5 = counters['correct_hit1'].value, counters['correct_hit3'].value, counters['correct_hit5'].value
        final_tot_ndcg = counters['total_ndcg5'].value
        final_fb_triggered = counters['total_feedback_triggered'].value
        final_rank_drops   = counters['total_rank_drops'].value
        final_rank_improves = counters['total_rank_improves'].value
        
        # NDCG Boost statistics
        avg_ndcg_v1 = counters['total_ndcg_v1'].value / total if total > 0 else 0
        avg_ndcg_final = counters['total_ndcg_final'].value / total if total > 0 else 0
        
        # Hit Rate Boost statistics
        avg_hit1_v1 = counters['total_hit1_v1'].value / total if total > 0 else 0
        avg_hit3_v1 = counters['total_hit3_v1'].value / total if total > 0 else 0
        avg_hit5_v1 = counters['total_hit5_v1'].value / total if total > 0 else 0
        
        avg_hit1_final = final_hit1 / total if total > 0 else 0
        avg_hit3_final = final_hit3 / total if total > 0 else 0
        avg_hit5_final = final_hit5 / total if total > 0 else 0

        avg_moe_only_ndcg = counters['total_moe_only_ndcg'].value / total if total > 0 else 0
        moe_better = counters['moe_better_count'].value
        reranker_better = counters['reranker_better_count'].value
        equal = counters['equal_count'].value
        
        # Mean/Std statistics
        all_gates = np.array(counters['gate_vals']) if counters['gate_vals'] else np.zeros((0,3))
        all_scores = np.array(counters['score_vals']) if counters['score_vals'] else np.zeros((0,2))
        
        gate_mean = all_gates.mean(axis=0) if len(all_gates)>0 else [0,0,0]
        gate_std = all_gates.std(axis=0) if len(all_gates)>0 else [0,0,0]
        score_mean = all_scores.mean(axis=0) if len(all_scores)>0 else [0,0]
        score_std = all_scores.std(axis=0) if len(all_scores)>0 else [0,0]

        gate_logs_list = list(counters['gate_vals'])
        if gate_logs_list:
            df_gates = pd.DataFrame(gate_logs_list, columns=['seq_weight', 'gcn_weight', 'sem_weight'])
            csv_path = os.path.join(output_dir, f'gating_weights_{args.dataset}.csv')
            df_gates.to_csv(csv_path, index=False)
            print(f"Saved gating weights CSV to: {csv_path}")



    save_final_metrics(args, total, final_hit1, final_hit3, final_hit5, final_tot_ndcg)
    
    # Write aggregate statistics file

    summary_text = (
        "=============================================\n"
        "FEEDBACK LOOP & GATING STATISTICS\n"
        "=============================================\n"
        f"• NDCG Boost (V1 -> Final): {avg_ndcg_v1:.4f} -> {avg_ndcg_final:.4f} ({avg_ndcg_final - avg_ndcg_v1:+.4f})\n"
        f"• Hit@1 Boost (V1 -> Final): {avg_hit1_v1:.4f} -> {avg_hit1_final:.4f} ({avg_hit1_final - avg_hit1_v1:+.4f})\n"
        f"• Hit@3 Boost (V1 -> Final): {avg_hit3_v1:.4f} -> {avg_hit3_final:.4f} ({avg_hit3_final - avg_hit3_v1:+.4f})\n"
        f"• Hit@5 Boost (V1 -> Final): {avg_hit5_v1:.4f} -> {avg_hit5_final:.4f} ({avg_hit5_final - avg_hit5_v1:+.4f})\n"
        f"• 📉 Rank Drops : {final_rank_drops} | 📈 Rank Improves: {final_rank_improves}\n"
        f"•  Avg Gates: Seq={gate_mean[0]:.3f}, GCN={gate_mean[1]:.3f}, Sem={gate_mean[2]:.3f}\n"
        "=============================================\n"
        " MOE VS RERANKER (ROUND 1) STATISTICS\n"
        "=============================================\n"
        f"• Avg MoE Only NDCG: {avg_moe_only_ndcg:.4f}\n"
        f"• Avg Reranker NDCG (V1): {avg_ndcg_v1:.4f}\n"
        f"• Avg Boost by Reranker: {avg_ndcg_v1 - avg_moe_only_ndcg:+.4f}\n"
        f"• Cases where Reranker improved MoE: {reranker_better}\n"
        f"• Cases where Reranker worsened MoE: {moe_better}\n"
        f"• Cases with equal performance: {equal}\n"
        "=============================================\n"
    )
    stats_path = os.path.join(output_dir, 'moe_weights_scores_stats.txt')
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write("=== FEEDBACK LOOP PERFORMANCE ===\n")
        f.write(f"Total Triggered: {final_fb_triggered}/{total}\n")
        f.write(f"Average NDCG V1 (No Feedback): {avg_ndcg_v1:.4f}\n")
        f.write(f"Average NDCG Final (With FB):  {avg_ndcg_final:.4f}\n")
        f.write(f"NDCG Boost:                   {avg_ndcg_final - avg_ndcg_v1:+.4f}\n\n")
        f.write(f"Hit@1 V1: {avg_hit1_v1:.4f} | Final: {avg_hit1_final:.4f} | Boost: {avg_hit1_final - avg_hit1_v1:+.4f}\n")
        f.write(f"Hit@3 V1: {avg_hit3_v1:.4f} | Final: {avg_hit3_final:.4f} | Boost: {avg_hit3_final - avg_hit3_v1:+.4f}\n")
        f.write(f"Hit@5 V1: {avg_hit5_v1:.4f} | Final: {avg_hit5_final:.4f} | Boost: {avg_hit5_final - avg_hit5_v1:+.4f}\n\n")
        f.write("=== MOE GATING WEIGHTS (MEAN ± STD) ===\n")
        f.write(f"Seq Gate: {gate_mean[0]:.4f} ± {gate_std[0]:.4f}\n")
        f.write(f"GCN Gate: {gate_mean[1]:.4f} ± {gate_std[1]:.4f}\n")
        f.write(f"Sem Gate: {gate_mean[2]:.4f} ± {gate_std[2]:.4f}\n\n")
        f.write("=== COMPONENT SCORES (MEAN ± STD) ===\n")
        f.write(f"MoE s0 (Fused): {score_mean[0]:.4f} ± {score_std[0]:.4f}\n")
        f.write(f"LLM Rerank Score: {score_mean[1]:.4f} ± {score_std[1]:.4f}\n")
        f.write(f"{summary_text}\n")

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    args = get_args()
    random.seed(args.seed)
    main(args)