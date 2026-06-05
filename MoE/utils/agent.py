import os
import torch
import sys
import pandas as pd
import numpy as np
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import model

sys.modules['SASRecModules_ori'] = model

from utils.helper_function import write_jsonl, read_jsonl, api_request
from utils.model import SASRec
from utils.text_processing import build_review_history, extract_item_text, ITEM_FETCH_KEYS

class UserModelAgent:
    def __init__(self, args, mode='prior_rec', shared_sasrec=None):
        self.memory = []
        self.info_list = []
        self.args = args
        self.mode = mode
        self.load_prompt()

        if shared_sasrec is not None:
            self.model    = shared_sasrec['model']
            self.id2name  = shared_sasrec['id2name']
            self.name2id  = shared_sasrec['name2id']
            self.id2rawid = shared_sasrec['id2rawid']
            self.seq_size = shared_sasrec['seq_size']
            self.item_num = shared_sasrec['item_num']
            self.device   = shared_sasrec['device']
            print("[UserModelAgent] Reusing shared SASRec — skipping reload.")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.id2name  = dict()
            self.name2id  = dict()
            self.id2rawid = dict()
            self.build_id2name()
            self.load_model()

    def build_id2name(self):
        if any(x in self.args.data_dir for x in ['yelp', 'amazon', 'goodreads']):
            # Step 1: read id2rawid to get raw_id for each inner_id
            rawid_path = os.path.join(self.args.data_dir, 'id2rawid.txt')
            if os.path.exists(rawid_path):
                with open(rawid_path, 'r', encoding='utf-8') as f:
                    for l in f.readlines():
                        ll = l.strip('\n').split('::')
                        if len(ll) >= 2:
                            self.id2rawid[int(ll[0])] = ll[1].strip()
            else:
                print(f"[UserModelAgent] WARNING: id2rawid.txt not found at {rawid_path}. "
                      f"Run process_data.py to generate it.")

            # Step 2: read id2name and build unique names as "ItemName [raw_id]"
            # Consistent with MoERecAgent._init_shared_resources() to avoid name mismatch.
            item_path = os.path.join(self.args.data_dir, 'id2name.txt')
            with open(item_path, 'r', encoding='utf-8') as f:
                for l in f.readlines():
                    ll = l.strip('\n').split('::', 1)
                    if len(ll) < 2:
                        continue
                    cid = int(ll[0])
                    orig_name = ll[1].strip()
                    raw_id = self.id2rawid.get(cid, str(cid))
                    unique_name = f"{orig_name} [{raw_id}]"
                    self.id2name[cid] = unique_name
                    self.name2id[unique_name] = cid
        else:
            raise ValueError("Invalid data dir: {}".format(self.args.data_dir))

    def load_model(self):
        print(f"Loading model from {self.args.model_path}")
        data_directory = self.args.data_dir
        data_statis = pd.read_pickle(os.path.join(data_directory, 'data_statis.df'))
        self.seq_size = data_statis['seq_size'][0]
        self.item_num = data_statis['item_num'][0]

        checkpoint = torch.load(self.args.model_path, map_location=self.device, weights_only=False)
        hidden_size = checkpoint.get('hidden_size', getattr(self.args, 'hidden_size', 64)) if isinstance(checkpoint, dict) else getattr(self.args, 'hidden_size', 64)
        self.model = SASRec(hidden_size, self.item_num, self.seq_size, 0.1, self.device)
        self.model.to(self.device)

        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
        else:
            self.model = checkpoint

        self.model.eval()
        print("Load model weights success!")

    def load_prompt(self):
        if self.mode == 'prior_rec':
            if 'amazon_musical' in self.args.data_dir:
                from constant.amazon_musical_prior_model_prompt import user_system_prompt, user_user_prompt, user_memory_system_prompt, user_memory_user_prompt, user_build_memory, user_build_memory_2
            elif 'amazon_industrial' in self.args.data_dir:
                from constant.amazon_industrial_prior_model_prompt import user_system_prompt, user_user_prompt, user_memory_system_prompt, user_memory_user_prompt, user_build_memory, user_build_memory_2
            elif 'amazon' in self.args.data_dir:
                from constant.amazon_prior_model_prompt import user_system_prompt, user_user_prompt, user_memory_system_prompt, user_memory_user_prompt, user_build_memory, user_build_memory_2
            elif 'goodreads' in self.args.data_dir:
                from constant.goodreads_prior_model_prompt import user_system_prompt, user_user_prompt, user_memory_system_prompt, user_memory_user_prompt, user_build_memory, user_build_memory_2
            elif 'yelp' in self.args.data_dir:
                from constant.yelp_prior_model_prompt import user_system_prompt, user_user_prompt, user_memory_system_prompt, user_memory_user_prompt, user_build_memory, user_build_memory_2
            else:
                raise ValueError("Invalid dataset: {}".format(self.args.data_dir))
            self.user_system_prompt = user_system_prompt
            self.user_user_prompt = user_user_prompt
            self.user_memory_system_prompt = user_memory_system_prompt
            self.user_memory_user_prompt = user_memory_user_prompt
            self.user_build_memory = user_build_memory
            self.user_build_memory_2 = user_build_memory_2


    def act(self, data, reason=None, item_list=None, docstore_cache=None):
        if self.mode == 'prior_rec':
            enriched_seq_str = build_review_history(data, id2name=self.id2name)

            _ds = next(
                (d for d in ['yelp', 'amazon', 'goodreads'] if d in getattr(self.args, 'data_dir', '')),
                'amazon'
            )

            def enrich_candidates(items_input):
                if not items_input:
                    return "None"
                items = items_input if isinstance(items_input, list) \
                        else [i.strip() for i in str(items_input).split(',')]
                interaction_tool = data.get('interaction_tool')
                id2rawid = data.get('id2rawid', self.id2rawid)
                if not interaction_tool:
                    return ", ".join(items)

                enriched = []
                for idx, item in enumerate(items, 1):
                    inner_id = self.name2id.get(item)
                    raw_id   = id2rawid.get(inner_id) if inner_id is not None else None
                    info_dict = {'Target_Name': item}
                    if raw_id:
                        try:
                            fetched = interaction_tool.get_item(item_id=raw_id)
                            if fetched:
                                for k in ITEM_FETCH_KEYS:
                                    if k in fetched:
                                        info_dict[k] = fetched[k]
                        except Exception:
                            pass
                    detail_str = extract_item_text(info_dict, _ds)
                    if detail_str and detail_str != 'No additional info':
                        enriched.append(f'#{idx}: "{item}" \u2014 {detail_str}')
                    else:
                        enriched.append(f'#{idx}: "{item}"')
                return "\n".join(enriched)

            if item_list is not None:
                items_raw = item_list if isinstance(item_list, list) \
                            else [i.strip() for i in str(item_list).split(',')]
                enriched_item_list = enrich_candidates(items_raw)
            else:
                enriched_item_list = "None"

            if len(self.memory) == 0:
                system_prompt = self.user_system_prompt.format(enriched_seq_str)
                user_prompt = self.user_user_prompt.format(
                    enriched_item_list,
                    reason,
                )
            else:
                system_prompt = self.user_memory_system_prompt.format(enriched_seq_str)
                memory_str = '\n'.join(self.memory[-2:])
                user_prompt = self.user_memory_user_prompt.format(
                    memory_str,
                    enriched_item_list,
                    reason,
                )
            response = api_request(system_prompt, user_prompt, self.args)
            try:
                out_dir = os.path.dirname(getattr(self.args, 'output_file', ''))
                if out_dir:
                    os.makedirs(out_dir, exist_ok=True)
                    log_path = os.path.join(out_dir, "user_agent_prompts.txt")
                    round_num   = len(self.memory) + 1
                    round_label = "INITIAL ROUND" if len(self.memory) == 0 else f"FEEDBACK ROUND (memory={len(self.memory)})"
                    SEP  = "=" * 80
                    SEP2 = "-" * 80
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(f"\n{SEP}\n")
                        f.write(f"[USER AGENT] User ID: {data.get('id', 'Unknown')} | ROUND {round_num} | {round_label}\n")
                        f.write(f"{SEP}\n")
                        f.write(f"[SYSTEM PROMPT]\n{SEP2}\n{system_prompt}\n{SEP2}\n")
                        f.write(f"[USER PROMPT]\n{SEP2}\n{user_prompt}\n{SEP2}\n")
                        f.write(f"[USER AGENT RESPONSE]\n{SEP2}\n{response}\n{SEP2}\n")
                        f.write(f"{SEP}\n")
            except Exception as e:
                print(f"Error logging prompt: {e}")
            return response
        else:
            raise ValueError("Invalid mode: {}".format(self.mode))

    def pred_model(self, data, score):
        if len(self.memory) == 0:
            system_prompt = self.user_system_prompt.format(data['seq_str'])
            user_prompt = self.user_user_prompt.format(data['pred_item'], score)
        else:
            system_prompt = self.user_memory_system_prompt.format(data['seq_str'], data['cans_str'], '\n'.join(self.memory))
            user_prompt = self.user_memory_user_prompt.format(data['pred_item'], score)

        response = api_request(system_prompt, user_prompt, self.args)
        return response

    def build_memory(self, info):
        if info['user_reason'] is not None:
                return self.user_build_memory.format(info['epoch'], info['rec_item_list'], info['rec_reason'], info['user_reason'])
        else:
            return self.user_build_memory_2.format(info['epoch'], info['rec_item_list'], info['rec_reason'])

    def update_memory(self, info):
        self.info_list.append(info)
        new_memory = self.build_memory(info)
        self.memory.append(new_memory)
        print(f"\n[UserModelAgent] New Memory built:\n{new_memory}\n")

    def save_memory(self, path):
        write_jsonl(path, self.info_list)

    def load_memory(self, path):
        self.info_list = read_jsonl(path)
        self.memory = [self.build_memory(info) for info in self.info_list]

    def regenerate_prior(self, data):
        data['prior_answer'] = self.model_generate(
            data['seq'], data['len_seq'], data['cans']
        )

    def model_generate(self, seq, len_seq, candidates):
        seq_b = [seq]
        len_seq_b = [len_seq]
        states = np.array(seq_b)
        states = torch.LongTensor(states).to(self.device)
        prediction = self.model.forward_eval(states, np.array(len_seq_b))

        sampling_idx = [True] * self.item_num
        cans_num = len(candidates)
        for i in candidates:
            sampling_idx.__setitem__(i, False)
        sampling_idxs = torch.stack([torch.tensor(sampling_idx)], dim=0)
        prediction = prediction.cpu().detach().masked_fill(sampling_idxs, prediction.min().item() - 1)
        values, topK = prediction.topk(cans_num, dim=1, largest=True, sorted=True)
        topK = topK.numpy()[0]
        name_list = [self.id2name[id] for id in topK]
        len_ret = int(len(name_list) / 4)
        return ', '.join(name_list[:len_ret])

    def score(self, seq, len_seq, candidates):
        seq_b = [seq]
        len_seq_b = [len_seq]
        states = np.array(seq_b)
        states = torch.LongTensor(states).to(self.device)
        prediction = self.model.forward_eval(states, np.array(len_seq_b))

        sampling_idx = [True] * self.item_num
        cans_num = len(candidates)
        for i in candidates:
            sampling_idx.__setitem__(i, False)
        sampling_idxs = torch.stack([torch.tensor(sampling_idx)], dim=0)
        prediction = prediction.cpu().detach().masked_fill(sampling_idxs, prediction.min().item() - 1)
        values, topK = prediction.topk(cans_num, dim=1, largest=True, sorted=True)
        values = values.numpy()[0]
        topK = topK.numpy()[0]
        score_dict = {}
        for i in range(len(topK)):
            id = topK[i]
            score = values[i]
            name = self.id2name[id]
            score_dict[name] = score
        return score_dict