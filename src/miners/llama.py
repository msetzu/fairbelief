from typing import List, Tuple

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import pipeline, AutoTokenizer, LlamaForCausalLM, AutoModelForCausalLM

from miners.mine import Miner, MinerConfig


class LLAMAMiner(Miner):
    """
    Miner for LLAMA models.
    """
    def __init__(self, model: str = "meta-llama/Llama-2-7b-hf", family: str = "bloom", device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
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

        model_kwargs = {
            "torch_dtype": dtype,
            "device_map": "balanced_low_0"
        }
        # self.pipeline = LlamaForCausalLM.from_pretrained(model, **model_kwargs)
        self.model = AutoModelForCausalLM.from_pretrained(model, **model_kwargs)
        self.pipeline = self.model
        # self.pipeline = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, device=0)

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
        for i in tqdm(config["indexes"]):
            # breakpoint()
            # config["template_column"] is 'maked_sentence'
            # config["original_mask"] is '[MASK]'
            input_sentence = prompts[i][config["template_column"]].replace(config["original_mask"], "")[:-1]
            max_length = len(self.tokenizer(input_sentence)["input_ids"]) + 10

            if self.do_rstrip is True:
                input_sentence = input_sentence.rstrip()

            with torch.inference_mode():
                inputs = self.tokenizer(input_sentence, return_tensors="pt")
                generations = self.pipeline.generate(inputs.input_ids, do_sample=False, num_beams=config["K"], num_return_sequences=config["K"],  max_length=max_length)
                model_predictions = self.tokenizer.batch_decode(generations, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                # model_predictions = self.pipeline(input_sentence, do_sample=False, num_beams=config["K"], num_return_sequences=config["K"], max_length=max_length)

            predictions = list()
            for p in model_predictions:
                # predictions.append(p["generated_text"][len(input_sentence) + 1:].split(" ")[0].replace(",", "").replace(".", "").replace("!", "").replace("?", ""))
                # predictions.append(p[len(input_sentence) + 1:])
                predictions.append(p["generated_text"][len(input_sentence) + 1:])

            # breakpoint()

            mine_results.append((i,
                                 prompts[i]["uuid"] if "uuid" in prompts[i] else "",
                                 input_sentence,
                                 predictions,
                                 prompts[i][config["fillin_value"]]))

        return mine_results
