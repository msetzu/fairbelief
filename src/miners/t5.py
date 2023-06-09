from typing import List, Tuple

import torch
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import pipeline, AutoTokenizer
from tqdm import tqdm

from miners.mine import Miner, MinerConfig


class T5Miner(Miner):
    """
    Miner for BLOOM models..
    """

    def __init__(self, model: str = "t5-small", family: str = "bloom", device: str = "cuda"):
        """
        Args:
            model: The model name (one of Huggingface's models)
            device: Device to load the model on, either "cuda" or "cpu"
        """
        super().__init__()

        self.do_rstrip = False
        if '-rstrip' in model:
            model = model.replace('-rstrip', '')
            self.do_rstrip = True

        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.pipeline = pipeline("text-generation", model=model, tokenizer=self.tokenizer, device=device)
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
            # breakpoint()
            # config["template_column"] is 'maked_sentence'
            # config["original_mask"] is '[MASK]'
            input_sentence = prompts[i][config["template_column"]].replace(config["original_mask"], "")[:-1]
            max_length = len(self.tokenizer(input_sentence)["input_ids"]) + 10

            if self.do_rstrip is True:
                input_sentence = input_sentence.rstrip()

            with torch.inference_mode():
                model_predictions = self.pipeline(input_sentence, do_sample=False, num_beams=config["K"], num_return_sequences=config["K"], max_length=max_length)

            predictions = list()
            for p in model_predictions:
                predictions.append(p["generated_text"][len(input_sentence) + 1:])

            # breakpoint()

            mine_results.append((i,
                                 prompts[i]["uuid"] if "uuid" in prompts[i] else "",
                                 input_sentence,
                                 predictions,
                                 prompts[i][config["fillin_value"]]))

            breakpoint()

        return mine_results


class T5MinerOld(Miner):
    """
    Miner for T5 models.
    """
    def __init__(self, model: str = "t5-small", device: str = "cuda"):
        """
        Args:
            model: The model name (one of Huggingface's T5 models)
            device: Device to load the model on, either "cuda" or "cpu"
        """
        super().__init__()
        self.tokenizer = T5Tokenizer.from_pretrained(model)
        self.model = T5ForConditionalGeneration.from_pretrained(model).to(device)
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
        for i in tqdm(config["indexes"]):
            if config["triples"]:
                input_sentence = prompts[i][config["template_column"]].replace(config["object_template"], config["mask_template"])
                input_sentence = input_sentence.replace(config["subject_template"], prompts[i][config["subject_value"]])
                tokenized_input = self.tokenizer(input_sentence)
            else:
                # adjust mask token
                input_sentence = prompts[i][config["template_column"]].replace(config["original_mask"], config["object_template"])
                # tokenize
                tokenized_input = self.tokenizer(input_sentence)
            # cut long text around the mask
            if len(tokenized_input["input_ids"]) > self.max_length:
                try:
                    mask_index = input_sentence.index(config["mask_template"])
                except ValueError as e:
                    print(f"error on {i}")
                    continue
                input_sentence = input_sentence[max(0, mask_index - 100):min(mask_index + 100, len(input_sentence))]
            inputs = self.tokenizer(input_sentence, return_tensors="pt", truncation=True, max_length=self.max_length).to(self.device)

            with torch.inference_mode():
                outputs = self.model.generate(inputs.input_ids, max_new_tokens=12)
                mine_results.append((i,
                                    prompts[i]["uuid"] if "uuid" in prompts[i] else "",
                                    input_sentence,
                                    self.tokenizer.decode(outputs[0], skip_special_tokens=True),
                                    prompts[i][config["fillin_value"]]))

        return mine_results
