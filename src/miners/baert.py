from typing import List, Tuple

import torch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaForMaskedLM, AutoModelForMaskedLM, DebertaTokenizer, BartTokenizer
from tqdm import tqdm

from miners.mine import Miner, MinerConfig


class BaertMiner(Miner):
    """
    Miner for Bert and Bart models (RoBERTa, DeBERTa, BART).
    """
    def __init__(self, model: str = "roberta-base", family: str = "roberta", device: str = "cuda"):
        """
        Args:
            model: The model name (one of Huggingface's models)
            family: One of "roberta", "deberta", "bart", according to the model
            device: Device to load the model on, either "cuda" or "cpu"
        """
        super().__init__()
        if family == "roberta":
            self.tokenizer = RobertaTokenizer.from_pretrained(model)
            self.model = RobertaForMaskedLM.from_pretrained(model).to(device)
        elif family == "deberta":
            self.model = AutoModelForMaskedLM.from_pretrained(model).to(device)
            self.tokenizer = DebertaTokenizer.from_pretrained(model)
        elif family == "bart":
            self.model = AutoModelForMaskedLM.from_pretrained(model).to(device)
        self.tokenizer = BartTokenizer.from_pretrained(model)
        self.max_length = 512
        self.device = device

    def mine(self, prompts: DataLoader, config: MinerConfig) -> List[Tuple[int, List[str], str]]:
        """
        Mine beliefs with the given `prompts`.

        Args:
            prompts: The dataset to mine
            config: Miner configuration. See `miner.MinerConfig`
        Returns:
            A list of tuples (index, top-k model predictions, ground truth prediction)
        """
        predictions = list()
        # mine
        for i in tqdm(config.indexes):
            if config.triples:
                input_sentence = prompts[i][config.template_column].replace(config.object_template, config.mask_template)
                input_sentence = input_sentence.replace(config.subject_template, prompts[i][config.subject_value])
                tokenized_input = self.tokenizer(input_sentence)
            else:
                input_sentence = prompts[i][config.template_column].replace("[MASK]", config.mask_template)
                tokenized_input = self.tokenizer(input_sentence)
            # cut long text around the mask
            if len(tokenized_input["input_ids"]) > self.max_length:
                mask_index = input_sentence.index(config.mask_template)
                input_sentence = input_sentence[max(0, mask_index - 100):min(mask_index + 100, len(input_sentence))]

            inputs = self.tokenizer(input_sentence, return_tensors="pt", truncation=True, max_length=self.max_length).to(self.device)
            with torch.no_grad():
                logits = self.model(**inputs)[0]
            mask_token_index = (inputs.input_ids == self.tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
            predicted_tokens = [logits[0, mask_token_index][0].argsort(descending=True)[k] for k in range(K)]
            predictions = [self.tokenizer.decode(token) for token in predicted_tokens]
            predictions = [p[1:] if p[0] == " " else p for p in predictions]

            predictions.append((i, predictions, prompts[i][config.fillin_value]))

        return predictions
