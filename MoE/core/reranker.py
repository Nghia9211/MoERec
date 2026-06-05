import json
import re
import threading
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel
import numpy as np
import math

from utils.text_processing import (
    build_review_history,
    extract_item_text,
    ITEM_FETCH_KEYS,
)

try:
    import tiktoken as _tiktoken
    _TIKTOKEN_ENC = _tiktoken.get_encoding("cl100k_base")
except Exception:
    _TIKTOKEN_ENC = None

_item_cache: Dict[str, dict] = {}
_item_cache_lock = threading.Lock()

def _cached_get_item(tool, raw_id: str):
    """Fetch item metadata with cache to avoid redundant lookups for the same raw_id."""
    with _item_cache_lock:
        if raw_id in _item_cache:
            return _item_cache[raw_id]
    result = tool.get_item(item_id=raw_id)
    with _item_cache_lock:
        _item_cache[raw_id] = result
    return result

# Structured output schema for LLM reranker.
# The LLM returns 1-based indices to avoid item name-mismatch issues.
class _RankerOutput(BaseModel):
    ranked_indices: List[int]   # 1-based indices matching the ML ranking order
    explanation: str


def rank_to_score(ranked_list: List[str]) -> Dict[str, float]:
    return {item: 1.0 / math.log2(rank + 2) for rank, item in enumerate(ranked_list)}

def _build_user_query(
    data: dict,
    candidate_names: list = None,
    id2name: Dict[int, str] = None,
) -> str:
    return build_review_history(data, id2name=id2name)


def _embed_similarity(query: str, candidate_names: List[str], candidate_texts: Dict[str, str], embedding_fn) -> Dict[str, float]:
    if not candidate_names or embedding_fn is None: return {n: 0.5 for n in candidate_names}
    try:
        from sklearn.metrics.pairwise import cosine_similarity as cos_sim
        query_vec  = embedding_fn.embed_query(query)
        texts      = [candidate_texts.get(n, n) for n in candidate_names]
        item_vecs  = embedding_fn.embed_documents(texts)
        sims       = cos_sim([query_vec], item_vecs)[0]
        min_s, max_s = float(sims.min()), float(sims.max())
        rng = max_s - min_s if max_s != min_s else 1.0
        return {n: float((sims[i] - min_s) / rng) for i, n in enumerate(candidate_names)}
    except Exception:
        return {n: 0.5 for n in candidate_names}

def _llm_rerank(
    llm, data: dict, name2id: Dict[str, int], candidate_names: List[str],
    user_query: str, memory: List[str], max_candidates: int = 10,
    output_dir: str = None,
    _QUERY_TOKEN_CAP: int = 3000,  
    _ITEMS_TOKEN_CAP: int = 4000,   
) -> Tuple[List[str], str]:
    cans_to_rank = candidate_names[:max_candidates]
    dataset = data.get('dataset') or next(
        (d for d in ['yelp', 'amazon', 'goodreads']
         if d in str(data.get('output_dir', ''))),
        'amazon'
    )
    task_type = {
        "goodreads": "Goodreads", "yelp": "Yelp", "amazon": "Amazon",
        "amazon_musical": "Amazon Musical Instruments",
        "amazon_industrial": "Amazon Industrial & Scientific",
    }.get(dataset, "Amazon")
    task_item = {
        "goodreads": "book", "yelp": "business", "amazon": "product",
        "amazon_musical": "musical instrument or accessory",
        "amazon_industrial": "industrial or scientific product",
    }.get(dataset, "product")

    interaction_tool = data.get('interaction_tool')
    id2rawid         = data.get('id2rawid', {})
    item_list_info   = []

    if interaction_tool and name2id and id2rawid:
        for name in cans_to_rank:
            inner_id  = name2id.get(name)
            raw_id    = id2rawid.get(inner_id)
            info_dict = {'Target_Name': name}
            if raw_id:
                try:
                    fetched = _cached_get_item(interaction_tool, raw_id)
                    if fetched:
                        for k in ITEM_FETCH_KEYS:
                            if k in fetched:
                                info_dict[k] = fetched[k]
                except Exception:
                    pass
            item_list_info.append(info_dict)
    else:
        item_list_info = [{'Target_Name': n} for n in cans_to_rank]

    numbered_lines = [
        f'  #{idx}: "{info.get("Target_Name", "Unknown")}" — {extract_item_text(info, dataset)}'
        for idx, info in enumerate(item_list_info, 1)
    ]
    ranked_display = "\n".join(numbered_lines)

    try:
        if _TIKTOKEN_ENC is not None:
            encoded_items = _TIKTOKEN_ENC.encode(ranked_display)
            if len(encoded_items) > _ITEMS_TOKEN_CAP:
                ranked_display = _TIKTOKEN_ENC.decode(encoded_items[:_ITEMS_TOKEN_CAP])
        else:
            ranked_display = ranked_display[:(_ITEMS_TOKEN_CAP * 4)]
    except Exception:
        ranked_display = ranked_display[:(_ITEMS_TOKEN_CAP * 4)]

    try:
        if _TIKTOKEN_ENC is not None:
            encoded_q = _TIKTOKEN_ENC.encode(user_query)
            if len(encoded_q) > _QUERY_TOKEN_CAP:
                user_query = _TIKTOKEN_ENC.decode(encoded_q[:_QUERY_TOKEN_CAP]) + "\n[... history truncated ...]"
        else:
            user_query = user_query[:(_QUERY_TOKEN_CAP * 4)]
    except Exception:
        user_query = user_query[:(_QUERY_TOKEN_CAP * 4)]

    if len(memory) > 0:
        dialogue_hist = "\n".join(memory[-2:])

        positive_items_text = ""
        pos_match = re.search(
            r'POSITIVE\s*MATCHES?\s*:(.*?)(?=\n\s*\d+\.\s*NEGATIVE|\Z)',
            dialogue_hist,
            re.IGNORECASE | re.DOTALL
        )
        if pos_match:
            pos_text = pos_match.group(1).strip()
            if pos_text and not re.match(r'^none\b', pos_text, re.IGNORECASE):
                positive_items_text = f"\nCRITICAL MUST-DO: The user explicitly identified the following as POSITIVE MATCHES: {pos_text}. You MUST place these items at the very TOP of your refined ranking (Rank 1 and/or Rank 2). Do NOT drop them, even if the user rejected the overall list."

        prompt = f"""You are a recommendation refinement system for {task_item}s on {task_type}.
A specialized ML model ranked candidates for this user. The user rejected the previous recommendation.
Your job is to REFINE the ranking based on the user's feedback.

User's Profile & History:
{user_query}

Previous Dialogue & User's Critique:
{dialogue_hist}

ML Model's Current Ranking (most recommended first):
{ranked_display}

REFINEMENT INSTRUCTIONS:{positive_items_text}
1. CRITICAL: Keep POSITIVE MATCHES (items the user explicitly praised) near the TOP. Any item identified as a POSITIVE MATCH in the critique MUST be promoted to Rank 1 or Rank 2.
2. Push DOWN items the user explicitly rejected as negative noise or a clear mismatch.
3. For items not mentioned: use the user's review history and item descriptions to determine the best order.
4. Output ranked_indices as the item #N numbers from the ML ranking above, ordered from best to worst.
   Example: if #3 is best, then #1, then #2 → ranked_indices: [3, 1, 2, ...]. Include all {len(cans_to_rank)} indices.
5. Provide a brief explanation of your changes (≤30 words)."""

    else:
        # ── FEW-SHOT EXAMPLE (compact, ~250 tokens, platform-specific) ──────
        _few_shot_by_ds = {
            "goodreads": f"""\
=== FORMAT EXAMPLE (fictional — for format guidance only) ===
User history (PAST reads — NOT candidates):
  [{{'item_name': 'Harry Potter [hp1]', 'stars': 5, 'text': 'Amazing fantasy!'}}]
ML Ranking:
  #1: "Twilight [tw1]" — rating: 3.8, shelves: romance
  #2: "The Hobbit [ho1]" — rating: 4.7, shelves: fantasy, adventure
  #3: "Cooking Basics [cb1]" — rating: 4.0, shelves: cooking
→ Swap justified (user history: strong fantasy preference):
  ranked_indices: [2, 1, 3]  explanation: "Promoted #2: user clearly prefers fantasy (5★). #3 kept last."
→ No swap needed (ML already correct):
  ranked_indices: [1, 2, 3]  explanation: "ML ranking preserved."
=== END EXAMPLE ===
⚠️  Review History = PAST items already read. Candidates = ONLY #1–#{len(cans_to_rank)} below.
To keep ML order exactly → ranked_indices: [1, 2, ..., {len(cans_to_rank)}].""",

            "amazon": f"""\
=== FORMAT EXAMPLE (fictional — for format guidance only) ===
User history (PAST purchases — NOT candidates):
  [{{'item_name': 'Wireless Headphones [B001]', 'stars': 5, 'text': 'Great sound quality!'}}]
ML Ranking:
  #1: "Bluetooth Speaker [B002]" — rating: 4.1, categories: Electronics
  #2: "Running Shoes [B003]" — rating: 4.5, categories: Sports
  #3: "USB-C Headphones [B004]" — rating: 4.6, categories: Electronics, Audio
→ Swap justified (user: electronics/audio preference):
  ranked_indices: [3, 1, 2]  explanation: "Promoted #3: strong audio preference from history. #2 (Sports) kept last."
→ No swap needed:
  ranked_indices: [1, 2, 3]  explanation: "ML ranking preserved."
=== END EXAMPLE ===
⚠️  Review History = PAST purchases. Candidates = ONLY #1–#{len(cans_to_rank)} below.
To keep ML order exactly → ranked_indices: [1, 2, ..., {len(cans_to_rank)}].""",

            "yelp": f"""\
=== FORMAT EXAMPLE (fictional — for format guidance only) ===
User history (PAST visits — NOT candidates):
  [{{'item_name': 'Pho Saigon [ys1]', 'stars': 5, 'text': 'Best Vietnamese food!'}}]
ML Ranking:
  #1: "Burger Palace [yb1]" — stars: 3.8, categories: Fast Food
  #2: "Pho Hanoi [yp1]" — stars: 4.7, categories: Vietnamese, Asian
  #3: "Sushi World [ys2]" — stars: 4.2, categories: Japanese
→ Swap justified (user: Vietnamese food preference):
  ranked_indices: [2, 3, 1]  explanation: "Promoted #2: user clearly prefers Vietnamese (5★)."
→ No swap needed:
  ranked_indices: [1, 2, 3]  explanation: "ML ranking preserved."
=== END EXAMPLE ===
⚠️  Review History = PAST visits. Candidates = ONLY #1–#{len(cans_to_rank)} below.
To keep ML order exactly → ranked_indices: [1, 2, ..., {len(cans_to_rank)}].""",
        }
        _few_shot_block = _few_shot_by_ds.get(dataset, _few_shot_by_ds["amazon"])

        prompt = f"""You are a recommendation refinement system for {task_item}s on {task_type}.
A specialized ML recommendation model has already ranked candidate {task_item}s for this user using multiple signals (sequential behavior patterns, collaborative filtering, and content similarity). Your job is to REFINE this ranking — not rebuild it from scratch.
The ML ranking is statistically reliable. Make MINIMAL adjustments only when clearly justified.

{_few_shot_block}

User's Profile & Review History:
{user_query}

ML Model's Ranking (most recommended → least recommended):
{ranked_display}

REFINEMENT INSTRUCTIONS:
1. CRITICAL: The ML model is statistically highly accurate. You must DEFAULT to preserving the ML ranking.
2. DO NOT demote the ML's #1 or #2 items unless they completely contradict the user's explicit preferences.
3. Only swap items if you are HIGHLY CONFIDENT (>90%) based on concrete proof in the review text. If unsure, output the original ML order.
4. Key signals: category/genre alignment, rating patterns, specific features the user mentions in reviews.
5. Output ranked_indices as the item #N numbers from the ML ranking above, ordered from best to worst.
   Example: if #3 is best, then #1, then #2 → ranked_indices: [3, 1, 2, ...]. Include all {len(cans_to_rank)} indices.
6. Provide a brief explanation of your changes (≤30 words), or 'ML ranking preserved' if no changes were needed."""

    # ── DIALOGUE LOGGING TO FILE (for paper) ──
    import os
    import threading
    _log_dir  = output_dir or data.get('output_dir', os.path.join(os.path.dirname(__file__), 'output'))
    os.makedirs(_log_dir, exist_ok=True)
    _log_path = os.path.join(_log_dir, 'reranker_dialogue_log.txt')

    # Module-level lock: ensures each block (prompt or response) is written
    # atomically even when the pipeline processes multiple users in parallel.
    if not hasattr(_llm_rerank, '_file_lock'):
        _llm_rerank._file_lock = threading.Lock()

    round_label = ("FEEDBACK REFINEMENT ROUND" if len(memory) > 0 else "INITIAL RANKING ROUND")
    SEP  = "=" * 80
    SEP2 = "-" * 80

    def _log_block(lines):
        """Write a list of lines as one atomic file write, protected by a lock."""
        block = "\n".join(lines) + "\n"
        with _llm_rerank._file_lock:
            with open(_log_path, 'a', encoding='utf-8') as f:
                f.write(block)

    user_id_str = str(data.get('id', data.get('user_id', 'Unknown')))
    round_num   = len(memory) + 1   # Round 1 = initial, Round 2+ = feedback
    # Write the entire prompt block atomically in one lock-protected write
    _log_block([
        f"\n{SEP}",
        f"[RERANKER] User ID: {user_id_str} | ROUND {round_num} | {round_label} | Platform: {task_type} | Item type: {task_item}",
        SEP,
        f"[PROMPT → RERANKER] (Round {round_num})",
        SEP2,
        prompt,
        SEP2,
    ])

    content     = "(no response yet)"
    ranked      = list(cans_to_rank)
    explanation = "(no explanation)"
    try:
        _MAX_OUTPUT = 1024
        structured_llm  = llm.bind(max_tokens=_MAX_OUTPUT).with_structured_output(_RankerOutput)
        result          = structured_llm.invoke(prompt)
        raw_indices     = result.ranked_indices   # e.g. [3, 1, 2, 5, 4, ...]
        explanation     = result.explanation

        import re as _re_expl
        def _short_name(full_name: str) -> str:
            return _re_expl.sub(r'\s*\[[^\]]*\]\s*$', '', full_name).strip()

        def _replace_index_ref(m):
            n = int(m.group(1))
            if 1 <= n <= len(cans_to_rank):
                return f'"{_short_name(cans_to_rank[n - 1])}" (#{n})'
            return m.group(0)

        if explanation and explanation not in ('(no explanation)', 'ML ranking preserved.', 'ML ranking preserved'):
            explanation = _re_expl.sub(r'#(\d+)', _replace_index_ref, explanation)
        n = len(cans_to_rank)
        seen   = set()
        ranked = []
        for idx in raw_indices:
            if 1 <= idx <= n and idx not in seen:
                ranked.append(cans_to_rank[idx - 1])
                seen.add(idx)
        for i, name in enumerate(cans_to_rank, 1):
            if i not in seen:
                ranked.append(name)
        content = json.dumps(
            {"ranked_indices": raw_indices, "ranked_items": ranked, "explanation": explanation},
            ensure_ascii=False, indent=2,
        )

    except Exception as e_structured:
        err_type = type(e_structured).__name__
        err_msg  = str(e_structured)

        if 'LengthFinishReason' in err_type or 'LengthFinish' in err_msg:
            warn = (
                f"[RERANKER WARNING] structured_output failed ({err_type}: {err_msg}), "
                "keeping ML order"
            )
        elif 'Timeout' in err_type or 'timeout' in err_msg.lower():
            warn = (
                f"[RERANKER WARNING] structured_output failed ({err_type}: {err_msg}), "
                "keeping ML order  ← vLLM timeout; prompt may still be too long."
            )
        else:
            warn = f"[RERANKER WARNING] structured_output failed ({err_type}: {err_msg}), keeping ML order"

        _log_block([warn])

    # Warn if LLM returned out-of-range or missing indices
    n_valid = len([r for r in ranked if r in set(cans_to_rank)])
    incomplete_warning = ""
    if n_valid < len(cans_to_rank):
        incomplete_warning = (
            f"[WARNING] Index mapping yielded only {n_valid}/{len(cans_to_rank)} valid items. "
            f"Missing items appended in ML order."
        )

    # Write the entire response block atomically in one lock-protected write
    log_lines = [
        f"[RERANKER RESPONSE] User ID: {user_id_str} ",
        SEP2,
        content,
        SEP2,
        f"[PARSED] Explanation: {explanation}",
        f"[PARSED] Ranked list ({len(ranked)} items): {ranked}",
    ]
    if incomplete_warning:
        log_lines.append(incomplete_warning)
    log_lines.append(SEP + "\n")
    _log_block(log_lines)

    # ranked is complete (missing items were appended in the try block)
    if len(candidate_names) > max_candidates:
        ranked += candidate_names[max_candidates:]

    return ranked, explanation

class Reranker:
    def __init__(self, embedding_fn=None, llm=None, mode: str='embed_only', enabled: bool=True, top_llm: int=10, output_dir: str=None):
        self.embedding_fn, self.llm, self.mode, self.enabled, self.top_llm = embedding_fn, llm, mode, enabled, top_llm
        self.output_dir = output_dir

    def rerank(self, data: dict, c_m: List[str], id2name: Dict[int, str]=None, name2id: Dict[str, int]=None, memory: List[str]=None) -> Tuple[List[str], Dict[str, float], str]:
        if not self.enabled or not c_m: return c_m, rank_to_score(c_m), "Reranker disabled."
        memory = memory or []
        try:
            if self.mode == 'embed_only': return self._embed_rerank(data, c_m, memory, id2name)
            elif self.mode == 'llm': return self._llm_only_rerank(data, name2id, c_m, memory, id2name)
            else: return self._embed_rerank(data, c_m, memory, id2name)
        except Exception as e: return c_m, rank_to_score(c_m), f"Reranker error: {e}"

    def _embed_rerank(self, data: dict, c_m: List[str], memory: List[str], id2name: Dict[int, str]=None) -> Tuple[List[str], Dict[str, float], str]:
        query = _build_user_query(data, id2name=id2name)
        if not query: return c_m, rank_to_score(c_m), "No user query available."
        sim_scores = _embed_similarity(query, c_m, {n: n for n in c_m}, self.embedding_fn)
        ranked = sorted(c_m, key=lambda n: sim_scores.get(n, 0.0), reverse=True)
        return ranked, rank_to_score(ranked), "Reranked by embedding similarity."

    def _llm_only_rerank(self, data: dict, name2id: Dict[str, int], c_m: List[str], memory: List[str], id2name: Dict[int, str]=None) -> Tuple[List[str], Dict[str, float], str]:
        if self.llm is None: return self._embed_rerank(data, c_m, memory, id2name)
        query = _build_user_query(data, candidate_names=c_m, id2name=id2name)
        ranked, explanation = _llm_rerank(self.llm, data, name2id, c_m, query, memory, max_candidates=self.top_llm, output_dir=self.output_dir)
        return ranked, rank_to_score(ranked), explanation

    @classmethod
    def from_shared(cls, shared: dict, llm=None, mode: str='embed_only', enabled: bool=True, top_llm: int=10, output_dir: str=None) -> "Reranker":
        return cls(embedding_fn=shared.get('embedding_function'), llm=llm, mode=mode, enabled=enabled, top_llm=top_llm, output_dir=output_dir)