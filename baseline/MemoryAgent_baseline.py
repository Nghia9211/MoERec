import json
import re
import logging
import os
import argparse
import tiktoken
from dotenv import load_dotenv

# Thêm thư viện để thực hiện cache model
import sentence_transformers
from huggingface_hub.utils import disable_progress_bars

from websocietysimulator import Simulator
from websocietysimulator.agent import RecommendationAgent
from websocietysimulator.llm import LLMBase, OpenAILLM
from websocietysimulator.agent.modules.memory_modules import MemoryDILU
from websocietysimulator.agent.modules.reasoning_modules import ReasoningIO
from utils.llm_provider import add_llm_args, build_llm_from_args

from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()

# --- TẮT LOG VÀ CẢNH BÁO ĐA LUỒNG ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"
disable_progress_bars() # Tắt thanh tiến trình Loading weights...

# --- BẮT ĐẦU ĐOẠN CACHE MODEL EMBEDDING ---
_original_init = sentence_transformers.SentenceTransformer.__init__
_model_cache = {}

def _cached_init(self, model_name_or_path, *args, **kwargs):
    # Nếu model này chưa từng được load, tiến hành load và lưu vào cache
    if model_name_or_path not in _model_cache:
        _original_init(self, model_name_or_path, *args, **kwargs)
        _model_cache[model_name_or_path] = self.__dict__.copy()
    else:
        # Nếu đã load rồi, copy reference từ cache sang instance mới
        self.__dict__ = _model_cache[model_name_or_path]

# Ghi đè hàm init của thư viện
sentence_transformers.SentenceTransformer.__init__ = _cached_init
# --- KẾT THÚC ĐOẠN CACHE MODEL ---


def num_tokens_from_string(string: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    try:
        return len(encoding.encode(string))
    except:
        return 0


class MyRecommendationAgent(RecommendationAgent):
    def __init__(self, llm: LLMBase):
        super().__init__(llm=llm)
        self.memory = MemoryDILU(llm=self.llm)
        # Tái sử dụng self.memory thay vì khởi tạo lại MemoryDILU()
        self.reasoning = ReasoningIO(profile_type_prompt='', memory=self.memory, llm=self.llm)

    def workflow(self):
        plan = [
            {'description': 'First I need to find user information'},
            {'description': 'Next, I need to find item information'},
            {'description': 'Next, I need to find review information'},
        ]

        user, item_list, history_review, filtered_reviews = '', [], '', []
        for sub_task in plan:
            if 'user' in sub_task['description']:
                user = str(self.interaction_tool.get_user(user_id=self.task['user_id']))
                if num_tokens_from_string(user) > 1000:
                    enc = tiktoken.get_encoding("cl100k_base")
                    user = enc.decode(enc.encode(user)[:1000])

            elif 'item' in sub_task['description']:
                keys = ['item_id', 'name', 'stars', 'review_count', 'attributes', 'categories', 'hours', 'address', 'city', 'state', 'title', 'average_rating', 'description', 'ratings_count', 'authors', 'publication_year', 'similar_books', 'price', 'brand', 'sales_rank']
                for item_id in self.task['candidate_list']:
                    item = self.interaction_tool.get_item(item_id=item_id)
                    if item:
                        item_list.append({k: item[k] for k in keys if k in item})
                    else:
                        print(f"Warning: No data found for item_id: {item_id}. Skipping.")

            elif 'review' in sub_task['description']:
                all_reviews      = self.interaction_tool.get_reviews(user_id=self.task['user_id'])
                candidate_ids    = set(self.task['candidate_list'])
                filtered_reviews = [r for r in all_reviews if r.get('item_id') not in candidate_ids]
                history_review = str(filtered_reviews[-15:]) 
                if num_tokens_from_string(history_review) > 8000:
                    enc = tiktoken.get_encoding("cl100k_base")
                    history_review = enc.decode(enc.encode(history_review)[:8000])

        retrieved_memory = ''
        if filtered_reviews:
            for his in filtered_reviews[-15:]:
                self.memory.addMemory(str(his))
            retrieved_memory = self.memory.retriveMemory(f"History review of {self.task['user_id']}")
        
        if num_tokens_from_string(retrieved_memory) > 8000:
            enc = tiktoken.get_encoding("cl100k_base")
            retrieved_memory = enc.decode(enc.encode(retrieved_memory)[:8000])

        task_description = f"""
You are a recommendation agent. Your task is to recommend items for a user based on their profile, historical reviews, and a list of candidate items.

--- CANDIDATE ITEMS ---
{self.task['candidate_list']}

--- ITEMS DESCRIPTION ---
{item_list}

--- PAST EXPERIENCE (Retrieved from Memory) ---
{retrieved_memory}

--- YOUR TASK ---
Based on all the information above, analyze the user's preferences and the attributes of the candidate items.
Your final output MUST BE list of strings, where each string is an item_id from the candidate list.
The list should be ranked from the most recommended to the least recommended item for this user.
Do not include any other text, explanations, or markdown formatting around the list.
The correct output format: [Sorted Candidate Item List]
""".strip()

        result = self.reasoning(task_description)
        print('Meta Output:', result)

        try:
            match = re.search(r"\[.*\]", result, re.DOTALL)
            result = match.group() if match else ''
            final  = eval(result)
            print('Processed Output:', final)
            return final
        except:
            print('format error')
            return ['']


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run WebSocietySimulator with Memory Agent")
    parser.add_argument('--task_set', default='amazon', choices=['amazon', 'yelp', 'goodreads'])
    parser.add_argument('--scenario', default='classic', choices=['classic', 'user_cold_start', 'item_cold_start'])
    add_llm_args(parser)
    args = parser.parse_args()

    task_set = args.task_set
    scenario = args.scenario

    load_dotenv()
    llm = OpenAILLM(
        api_key="EMPTY", 
        model="qwen-small", 
        base_url="http://localhost:8036/v1"
    )

    simulator = Simulator(data_dir="../dataset/output_data_all/", device="gpu", cache=True)
    simulator.set_task_and_groundtruth(
        task_dir=f"../dataset/tasks5/{scenario}/{task_set}/tasks",
        groundtruth_dir=f"../dataset/tasks5/{scenario}/{task_set}/groundtruth",
    )
    simulator.set_agent(MyRecommendationAgent)
    simulator.set_llm(llm)

    agent_outputs      = simulator.run_simulation(number_of_tasks=None, enable_threading=True, max_workers=15)
    evaluation_results = simulator.evaluate()

    os.makedirs(f'./results/{scenario}', exist_ok=True)
    with open(f'./results/{scenario}/evaluation_results_MemoryAgent_{task_set}.json', 'w') as f:
        json.dump(evaluation_results, f, indent=4)
    print(f"The evaluation_results for {task_set} is: {evaluation_results}")