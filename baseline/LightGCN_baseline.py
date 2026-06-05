"""
LightGCN Standalone Baseline
============================
Evaluates a pre-trained LightGCN model through the WebSocietySimulator
framework, following the same pattern as SimpleX_baseline.py.

Workflow:
  1. Load pre-trained node (user/item) embeddings from plugin/lightGCN/gcn_embedding/
  2. For each task, rank the candidate_list by cosine similarity to the user embedding
  3. Evaluate via Simulator → HR@1, HR@3, HR@5, NDCG@5

Usage:
    cd baseline/
    python LightGCN_baseline.py --task_set yelp --scenario classic
    python LightGCN_baseline.py --task_set amazon --scenario classic
    python LightGCN_baseline.py --task_set amazon --scenario user_cold_start
    python LightGCN_baseline.py --task_set goodreads --scenario classic
"""

import argparse
import json
import logging
import os
import sys

import torch
import torch.nn.functional as F

from websocietysimulator import Simulator
from websocietysimulator.agent import RecommendationAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LightGCN Recommendation Agent
# ---------------------------------------------------------------------------

class LightGCNRecommendationAgent(RecommendationAgent):
    """
    Pure CF agent using pre-trained LightGCN embeddings.
    No LLM required. Ranks candidates by cosine similarity.
    """
    _node_embs: dict = {}   # { node_id_str: tensor }
    _dataset: str = ""

    @classmethod
    def load_embeddings(cls, emb_dir: str, dataset: str):
        """
        Load node embeddings from disk.
        Called once before Simulator.run_simulation().

        Args:
            emb_dir : path to the root embeddings directory
                      (e.g. '../plugin/lightGCN/gcn_embedding')
            dataset : 'amazon' | 'yelp' | 'goodreads'
        """
        emb_path = os.path.join(emb_dir, f"{dataset}_gcn_emb.pt")

        if not os.path.exists(emb_path):
            if dataset == "amazon":
                fallback_path = os.path.join(emb_dir, "amazon_musical_gcn_emb.pt")
                if os.path.exists(fallback_path):
                    logger.info(f"Fallback to {fallback_path} for dataset 'amazon'")
                    emb_path = fallback_path
                else:
                    raise FileNotFoundError(
                        f"Embeddings not found in '{emb_path}' or '{fallback_path}'.\n"
                    )
            else:
                raise FileNotFoundError(
                    f"Embeddings not found at '{emb_path}'.\n"
                    f"Please check the path or run LightGCN training first."
                )

        logger.info(f"Loading LightGCN embeddings from {emb_path} …")
        cls._node_embs = torch.load(emb_path, map_location="cpu", weights_only=False)
        cls._dataset   = dataset
        logger.info(
            f"  Nodes loaded: {len(cls._node_embs):,}"
        )

    def __init__(self, llm=None):
        """llm is accepted but ignored — LightGCN doesn't need an LLM."""
        super().__init__(llm=llm)

    def workflow(self):
        """
        Rank the candidate_list by cosine similarity to the user embedding.

        Returns:
            List of item_id strings, ranked most → least relevant.
            Falls back to the original candidate_list order if the user is
            unknown (cold-start scenario).
        """
        user_id    = self.task["user_id"]
        candidates = self.task["candidate_list"] 

        # ---- Look up user embedding ----------------------------------------
        u_emb = self._node_embs.get(user_id)
        if u_emb is None:
            # Unknown user (user cold-start) → return original order
            logger.debug(f"Unknown user '{user_id}', returning original order.")
            return candidates

        u_emb = F.normalize(u_emb.unsqueeze(0), dim=-1)  # (1, d)

        # ---- Collect item embeddings for candidates ------------------------
        scores = []
        for item_id in candidates:
            i_emb = self._node_embs.get(item_id)
            if i_emb is None:
                # Unknown item → score of -1 (will be ranked last)
                scores.append(-1.0)
            else:
                i_emb = F.normalize(i_emb.unsqueeze(0), dim=-1)  # (1, d)
                score = (u_emb * i_emb).sum().item()
                scores.append(score)

        # ---- Sort candidates by score descending ---------------------------
        ranked = [item_id for _, item_id in
                  sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)]

        return ranked


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run LightGCN Baseline via WebSocietySimulator")
    parser.add_argument(
        "--task_set", default="yelp",
        choices=["amazon", "yelp", "goodreads", "amazon_musical", "amazon_industrial"],
        help="Dataset to evaluate on"
    )
    parser.add_argument(
        "--scenario", default="classic",
        choices=["classic", "user_cold_start", "item_cold_start"],
        help="Evaluation scenario"
    )
    parser.add_argument(
        "--data_dir", default="../dataset/output_data_all",
        help="Path to output_data_all directory"
    )
    parser.add_argument(
        "--emb_dir", default="../plugin/MoE/saved_models",
        help="Path to LightGCN embeddings directory"
    )
    parser.add_argument(
        "--tasks_dir", default="../dataset/tasks5",
        help="Path to tasks directory (contains classic/, user_cold_start/, …)"
    )
    def parse_num_tasks(value):
        if str(value).lower() == 'none':
            return None
        return int(value)

    parser.add_argument(
        "--num_tasks", type=parse_num_tasks, default=500,
        help="Number of tasks to evaluate (default: 500, use 'None' for all)"
    )
    parser.add_argument(
        "--max_workers", type=int, default=20,
        help="Number of parallel threads"
    )
    args = parser.parse_args()

    task_set = args.task_set
    scenario = args.scenario

    # ---- Load embeddings ---------------------------------------------------
    LightGCNRecommendationAgent.load_embeddings(args.emb_dir, task_set)

    # ---- Setup Simulator ---------------------------------------------------
    if task_set in ["amazon", "yelp", "goodreads"]:
        simulator = Simulator(
            data_dir=args.data_dir,
            device="cpu",  # LightGCN inference is CPU-based (embedding lookup)
            cache=True,
        )
    elif task_set == "amazon_musical":
        simulator = Simulator(
            data_dir="../dataset/musical_industrial/musical_amazon",
            device="cpu",
            cache=True,
        )
    elif task_set == "amazon_industrial":
        simulator = Simulator(
            data_dir="../dataset/musical_industrial/industrial_amazon",
            device="cpu",
            cache=True,
        )

    simulator.set_task_and_groundtruth(
        task_dir=os.path.join(args.tasks_dir, scenario, task_set, "tasks"),
        groundtruth_dir=os.path.join(args.tasks_dir, scenario, task_set, "groundtruth"),
    )
    simulator.set_agent(LightGCNRecommendationAgent)
    simulator.set_llm(None)

    # ---- Run ---------------------------------------------------------------
    logger.info(f"Running LightGCN baseline on {task_set} / {scenario} …")
    simulator.run_simulation(
        number_of_tasks=args.num_tasks,
        enable_threading=True,
        max_workers=args.max_workers,
    )
    evaluation_results = simulator.evaluate()

    # ---- Save results ------------------------------------------------------
    out_dir = os.path.join("results", scenario)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"evaluation_results_tGCN_{task_set}.json")

    with open(out_path, "w") as f:
        json.dump(evaluation_results, f, indent=4)

    logger.info(f"Results saved to {out_path}")
    print(f"\n=== LightGCN Evaluation Results ({task_set} / {scenario}) ===")
    print(json.dumps(evaluation_results, indent=4))


if __name__ == "__main__":
    main()
