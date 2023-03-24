from typing import List, Tuple

import torch
from torch.utils.data import DataLoader
from transformers import RealmTokenizer, RealmRetriever, RealmKnowledgeAugEncoder, RealmForOpenQA
from tqdm import tqdm

from miners.mine import Miner, MinerConfig


class RealmMiner(Miner):
    """
    Miner for Rag models.
    """
    def __init__(self, model: str = "google/realm-cc-news-pretrained-encoder", device: str = "cuda"):
        """
        Args:
            model: The model name (one of Huggingface's Realm models)
            device: Device to load the model on, either "cuda" or "cpu"
        """
        super().__init__()
        self.tokenizer = RealmTokenizer.from_pretrained(model)
        self.retriever = RealmRetriever.from_pretrained(model)
        self.model = RealmKnowledgeAugEncoder.from_pretrained("google/realm-cc-news-pretrained-encoder", num_candidates=2,
                                                              retriever=self.retriever).to(device)
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
        mine_results = list()
        for i in tqdm(config["indexes"]):
            if config["triples"]:
                input_sentence = prompts[i][config["template_column"]].replace(config["object_template"], config["mask_template"])
                input_sentence = input_sentence.replace(config["subject_template"], prompts[i][config["subject_value"]])
                tokenized_input = self.tokenizer(input_sentence)
            else:
                # adjust mask token
                input_sentence = prompts[i][config["template_column"]].replace(config["original_mask"], config["mask_template"])
                # tokenize
                tokenized_input = self.tokenizer(input_sentence)
            # cut long text around the mask
            if len(tokenized_input["input_ids"]) > self.max_length:
                mask_index = input_sentence.index(config["mask_template"])
                input_sentence = input_sentence[max(0, mask_index - 100):min(mask_index + 100, len(input_sentence))]
            inputs = self.tokenizer(input_sentence, return_tensors="pt", truncation=True, max_length=self.max_length).to(self.device)

            with torch.no_grad():
                logits = self.model(**inputs)[0]
            mask_token_index = (inputs.input_ids == self.tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
            predicted_tokens = [logits[0, mask_token_index][0].argsort(descending=True)[k] for k in range(config["K"])]
            predictions = [self.tokenizer.decode(token) for token in predicted_tokens]
            # for some reason REALM introduces spaces between characters
            predictions = [p[1:].replace(" ", "") if p[0] == " " else p.replace(" ", "") for p in predictions]

            mine_results.append((i,
                                 prompts[i]["uuid"] if "uuid" in prompts[i] else "",
                                 predictions,
                                 prompts[i][config["fillin_value"]]))

        return mine_results


class RealmQAMiner(Miner):
    """
    Miner for Realm QA models.
    """
    def __init__(self, model: str = "google/realm-orqa-nq-openqa", device: str = "cuda"):
        """
        Args:
            model: The model name (one of Huggingface's RAG models)
            device: Device to load the model on, either "cuda" or "cpu"
        """
        super().__init__()
        self.tokenizer = RealmTokenizer.from_pretrained(model)
        self.retriever = RealmRetriever.from_pretrained(model)
        self.model = RealmForOpenQA.from_pretrained("google/realm-cc-news-pretrained-encoder", retriever=self.retriever).to(device)
        self.max_length = 512
        self.device = device

    def mine(self, prompts: DataLoader, **kwargs) -> List[Tuple[int, str, str, List[str], str]]:
        """
        Mine beliefs with the given `prompts`.

        Args:
            prompts: The dataset to mine. Should be a DataLoader with "masked_sentence" (holding the cloze-style
                    prompt) and "obj_surface" (the ground truth prediction) columns.
            **kwargs: Additional parameters:
                        with_context: True to use the available context, False otherwise. Defaults to False.

        Returns:
            A list of tuples (index, top-k model predictions, ground truth prediction)
        """
        predictions = list()
        with_context = kwargs.get("with_context", False)
        for i in tqdm(range(len(prompts))):
            # adjust mask token
            question = prompts[i]["question"]
            context = prompts[i]["context"]
            # tokenize
            tokenized_question = self.tokenizer(question)
            tokenized_context = self.tokenizer(context)
            inputs = self.tokenizer(context if with_context else "" + question, return_tensors="pt",
                                    truncation=True, max_length=self.max_length)
            answer_ids = self.tokenizer(prompts[i]["answers"],
                                        add_special_tokens=False,
                                        return_token_type_ids=False,
                                        return_attention_mask=False,).input_ids

            reader_output, predicted_answer_ids = self.model(**inputs, answer_ids=answer_ids, return_dict=False)
            predicted_answer = self.tokenizer.decode(predicted_answer_ids)

            predictions.append((i, prompts[i]["uuid"], context if with_context else "" + question, predicted_answer, prompts[i][
                "answers"]["text"]))

        return predictions
