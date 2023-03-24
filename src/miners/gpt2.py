from typing import List, Tuple

import torch
from torch.utils.data import DataLoader
from transformers import pipeline, AutoTokenizer

from miners.mine import Miner, MinerConfig


class GPT2Miner(Miner):
    """
    Miner for GPT2 models..
    """
    def __init__(self, model: str = "gpt2", family: str = "gpt2", device: str = "cuda"):
        """
        Args:
            model: The model name (one of Huggingface's models)
            device: Device to load the model on, either "cuda" or "cpu"
        """
        super().__init__()
        self.pipeline = pipeline("text-generation", model=model)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.max_length = 512
        self.device = device

    def mine(self, prompts: DataLoader, config: MinerConfig) -> List[Tuple[int, str, str, List[str], str]]:
        """
        Mine beliefs with the given `prompts`.

        Args:
            prompts: The dataset to mine
            config: Miner configuration. See `miner.MinerConfig`
        Returns:
            A list of tuples (index, top-k model predictions, ground truth prediction)
        """
        mine_results = list()
        # mine
        for i in config["indexes"]:
            input_sentence = prompts[i][config["template_column"]].replace(config["original_mask"], " ")[:-1]
            max_length = len(self.tokenizer(input_sentence)["input_ids"]) + 5
            model_predictions = self.pipeline(input_sentence, num_return_sequences=config["K"])
            
            predictions = list()
            for p in model_predictions:
                predictions.append(p["generated_text"][len(input_sentence) + 1:].split(" ")[0].replace(",", "").replace(".", "").replace("!", "").replace("?", ""))

            mine_results.append((i,
                                 prompts[i]["uuid"] if "uuid" in prompts[i] else "",
                                 input_sentence,
                                 predictions,
                                 prompts[i][config["fillin_value"]]))

        return mine_results
