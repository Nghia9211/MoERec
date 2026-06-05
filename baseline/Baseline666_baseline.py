import json
import os
import re
import logging
import argparse
import tiktoken
from dotenv import load_dotenv

from websocietysimulator import Simulator
from websocietysimulator.agent import RecommendationAgent
from websocietysimulator.llm import LLMBase, OpenAILLM 

def num_tokens_from_string(string: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    try:
        return len(encoding.encode(string))
    except:
        return 0

class RecReasoning:
    def __init__(self, llm):
        self.llm = llm

    def __call__(self, task_description: str):
        messages = [{"role": "user", "content": task_description}]
        return self.llm(messages=messages, temperature=0.1, max_tokens=500)

class MyRecommendationAgent(RecommendationAgent):
    def __init__(self, llm: LLMBase):
        super().__init__(llm=llm)
        self.reasoning = RecReasoning(llm=self.llm)

    def workflow(self):
        item_keys = ['item_id', 'name', 'title', 'stars', 'description','review_count','attributes', 'title', 'average_rating','rating_number', 'description', 'ratings_count', 'title_without_series' ]
        rev_keys  = ['item_id', 'rating', 'stars', 'text', 'timestamp', 'verified_purchase', 'title', 'helpful_vote']

        all_reviews = self.interaction_tool.get_reviews(user_id=self.task['user_id']) or []
        candidate_ids = set(self.task['candidate_list'])

        history_reviews_list = []
        for r in all_reviews:
            if r.get('item_id') not in candidate_ids:
                history_reviews_list.append({k: r.get(k) for k in rev_keys if k in r})

        history_str = str(history_reviews_list)
        if num_tokens_from_string(history_str) > 8000: 
            enc = tiktoken.get_encoding("cl100k_base")
            history_str = enc.decode(enc.encode(history_str)[:8000])

        item_details = []
        for item_id in self.task['candidate_list']:
            item = self.interaction_tool.get_item(item_id=item_id)
            item_details.append({k: item.get(k) for k in item_keys if k in item} if item else {'item_id': item_id})
        task_description = f"""
You are a real user on an online platform.
Your historical item review text and stars are as follows: {history_str}

Now you need to rank the following 20 items: {self.task['candidate_list']}
according to their match degree to your preference.

Please rank the more interested items more front in your rank list.
The information of the above 20 candidate items is as follows: {item_details}

Your final output should be ONLY a ranked item list of {self.task['candidate_list']}
with the following format!
DO NOT introduce any other item ids! DO NOT output your analysis process!
The correct output format: [Sorted Candidate Item List]
Output ONLY a ranked list of the candidate item IDs in this format: [ID1, ID2, ..., ID20]
        """.strip()

        # total_tokens = num_tokens_from_string(task_description)
        # if total_tokens > 8000:
        #     enc = tiktoken.get_encoding("cl100k_base")
        #     task_description = enc.decode(enc.encode(task_description)[:14000])
        result = self.reasoning(task_description)
        
        try:
            matches = re.findall(r"\[(.*?)\]", result, re.DOTALL)
            if matches:
                items = [i.strip().strip("'").strip('"') for i in matches[-1].split(',')]
                final_list = [i for i in items if i in candidate_ids]
                remaining = [i for i in self.task['candidate_list'] if i not in final_list]
                return (final_list + remaining)[:20]
        except:
            pass
        return self.task['candidate_list']

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_set', default='amazon', choices=['amazon', 'yelp', 'goodreads'])
    parser.add_argument('--scenario', default='classic', choices=['classic', 'user_cold_start', 'item_cold_start'])
    args = parser.parse_args()

    load_dotenv()
    task_set = args.task_set
    scenario = args.scenario


    llm = OpenAILLM(
    api_key="EMPTY", 
    model="qwen-research", 
    base_url="http://localhost:11435/v1"
    )

    # simulator = Simulator(data_dir="../dataset/output_data_all/", device="gpu", cache=True)
    # simulator.set_task_and_groundtruth(
    #     task_dir=f"../dataset/tasks5/{scenario}/{args.task_set}/tasks",
    #     groundtruth_dir=f"../dataset/tasks5/{scenario}/{args.task_set}/groundtruth",
    # )
    simulator = Simulator(data_dir="../1_5_video_games_amazon/", device="gpu", cache=True)
    simulator.set_task_and_groundtruth(
        task_dir=f"../1_5_video_games_amazon/task5_amazon_new/{scenario}/amazon/tasks",
        groundtruth_dir=f"../1_5_video_games_amazon/task5_amazon_new/{scenario}/amazon/groundtruth",
    )
    simulator.set_agent(MyRecommendationAgent)
    simulator.set_llm(llm)
    agent_outputs = simulator.run_simulation(number_of_tasks=None, enable_threading=True, max_workers=10)

    evaluation_results = simulator.evaluate()
    os.makedirs(f'./results/{scenario}', exist_ok=True)
    with open(f'./results/{scenario}/evaluation_results_baseline666_{task_set}_{dataset}.json', 'w') as f:
        json.dump(evaluation_results, f, indent=4)
    print(f"The evaluation_results for {task_set} is: {evaluation_results}")