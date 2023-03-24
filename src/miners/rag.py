from typing import List, Tuple, Optional
import sys
import os

sys.path.append("../")

import torch
from torch.utils.data import DataLoader
from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration, RagSequenceForGeneration
from tqdm import tqdm

from miners.mine import Miner, MinerConfig


class RagMiner(Miner):
    """
    Miner for Rag models.
    """
    def __init__(self, model: str = "facebook/rag-token-nq", rag_dataset: Optional[str] = "wiki_dpr", device: str = "cuda"):
        """
        Args:
            model: The model name (one of Huggingface's RAG models)
            device: Device to load the model on, either "cuda" or "cpu"
        """
        super().__init__()
        self.tokenizer = RagTokenizer.from_pretrained(model)
        self.retriever = RagRetriever.from_pretrained(model, index_name="exact",
                                                      use_dummy_dataset=True if rag_dataset is None else False,
                                                      dataset=rag_dataset)
        self.model = RagSequenceForGeneration.from_pretrained(model, retriever=self.retriever).to(device)
        self.max_length = 512
        self.device = device

    def mine(self, prompts: DataLoader, config: MinerConfig) -> List[Tuple[int, str, str, List[str], str, List[str]]]:
        """Mine beliefs with the given `prompts`.

        Args:
            prompts: The dataset to mine
            config: Miner configuration. See `miner.MinerConfig`

        Returns:
            A list of tuples (index, top-k model predictions, ground truth prediction)
        """
        mine_results = list()
        # construct prompts
        for i in tqdm(config["indexes"]):
            if config["triples"]:
                input_sentence = prompts[i][config["template_column"]].replace(config["object_template"], config["mask_template"])
                input_sentence = input_sentence.replace(config["subject_template"], prompts[i][config["subject_value"]])
                tokenized_input = self.tokenizer(input_sentence)
            else:
                input_sentence = prompts[i][config["template_column"]].replace(config["original_mask"], config["mask_template"])
                tokenized_input = self.tokenizer(input_sentence)
            # cut long text around the mask
            if len(tokenized_input["input_ids"]) > self.max_length:
                mask_index = input_sentence.index("<mask>")
                input_sentence = input_sentence[max(0, mask_index - 100):min(mask_index + 100, len(input_sentence))]
            inputs = self.tokenizer(input_sentence, return_tensors="pt", truncation=True, max_length=self.max_length).to(self.device)

            predictions = self.tokenizer.decode(self.model.generate(input_ids=inputs["input_ids"])[0]).replace("</s>", "")[1:]

            retriever_input_ids = self.model.retriever.question_encoder_tokenizer(input_sentence,
                                                                                  return_tensors="pt",
                                                                                  padding=True,
                                                                                  truncation=True)["input_ids"]
            question_enc_outputs = self.model.rag.question_encoder(retriever_input_ids.to("cuda"))
            question_enc_pool_output = question_enc_outputs[0]
            result = self.model.retriever(retriever_input_ids,
                                          question_enc_pool_output.cpu().detach().to(torch.float32).numpy(),
                                          prefix=self.model.rag.generator.config.prefix,
                                          n_docs=self.model.config.n_docs,
                                          return_tensors="pt")
            all_docs = self.model.retriever.index.get_doc_dicts(result.doc_ids)

            mine_results.append((i,
                                 prompts[i]["uuid"] if "uuid" in prompts[i] else "",
                                 input_sentence,
                                 [predictions],
                                 prompts[i][config["fillin_value"]],
                                 [self.__strip_title(doc) for doc in all_docs[0]["text"]]))

        return mine_results

    def __strip_title(self, title: str):
        if title.startswith('"'):
            title = title[1:]
        if title.endswith('"'):
            title = title[:-1]

        return title
