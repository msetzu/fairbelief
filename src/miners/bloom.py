from typing import List, Tuple

from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

from miners.mine import Miner, MinerConfig


class BLOOMMiner(Miner):
    """
    Miner for BLOOM models.
    """
    def __init__(self, model: str = "bigscience/bloom-1b1", device: str = "cuda"):
        """
        Args:
            model: The model name (one of BigScience's BLOOM models)
            device: Device to load the model on, either "cuda" or "cpu"
        """
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(model).to(device)
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

            outputs = self.model.generate(inputs.input_ids, max_new_tokens=12)
            mine_results.append((i,
                                 prompts[i]["uuid"] if "uuid" in prompts[i] else "",
                                 input_sentence,
                                 self.tokenizer.decode(outputs[0], skip_special_tokens=True),
                                 prompts[i][config["fillin_value"]]))

        return mine_results
