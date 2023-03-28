import copy
import json
import sys
import os
from typing import Optional


sys.path.append(os.path.abspath(".") + "/../")

# set huggingface cache folder, must be done before loading the module
# adjust to your liking
# os.environ["TRANSFORMERS_CACHE"] = "/home/mattiasetzu/disk1/huggingface_cache/"
# os.environ["HF_DATASETS_CACHE"] = "/home/mattiasetzu/disk1/huggingface_cache/"

import fire as fire
from datasets import load_dataset
import torch

# from preprocessing.crows import CrowsDataLoader
# from preprocessing.stereoset import StereosetDataLoader

from miners.baert import BaertMiner
from miners.mine import LAMA_BAERT_MINER as lama_baert_mining_config
from miners.mine import LAMA_DEBERTA_MINER as lama_deberta_mining_config
from miners.mine import LAMA_T5_MINER as lama_t5_mining_config
from miners.mine import LAMA_BLOOM_MINER as lama_bloom_mining_config
from miners.mine import LAMA_RAG_MINER as lama_rag_mining_config
from miners.mine import HONEST_BAERT_MINER as honest_baert_mining_config
from miners.mine import HONEST_DEBERTA_MINER as honest_deberta_mining_config
from miners.mine import HONEST_T5_MINER as honest_t5_mining_config
from miners.mine import HONEST_RAG_MINER as honest_rag_mining_config
from miners.mine import SQUAD_BAERT_MINER as squad_baert_mining_config
from miners.mine import SQUAD_DEBERTA_MINER as squad_deberta_mining_config
from miners.mine import SQUAD_T5_MINER as squad_t5_mining_config
from miners.mine import SQUAD_RAG_MINER as squad_rag_mining_config
from miners.mine import CROWS_BAERT_MINER as crows_baert_mining_config
from miners.mine import CROWS_DEBERTA_MINER as crows_deberta_mining_config
from miners.mine import CROWS_T5_MINER as crows_t5_mining_config
from miners.mine import CROWS_RAG_MINER as crows_rag_mining_config
from miners.mine import STEREOSET_BAERT_MINER as stereoset_baert_mining_config
from miners.mine import STEREOSET_DEBERTA_MINER as stereoset_deberta_mining_config
from miners.mine import STEREOSET_T5_MINER as stereoset_t5_mining_config
from miners.mine import STEREOSET_RAG_MINER as stereoset_rag_mining_config
from miners.gpt2 import GPT2Miner
from miners.t5 import T5Miner
from miners.rag import RagMiner
from miners.bloom import BLOOMMiner


device = "cuda" if torch.cuda.is_available() else "cpu"


def extract(dataset: str, model: str = "roberta-base", subset: Optional[str] = "trex", dump_file: str = "dump"):
    """Extract beliefs for the given `model` on the given `dataset[subset]`, and dump results in `dump_file`.

    Args:
        dataset (str): The dataset, e.g., "lama"
        model (str): The model, e.g., "roberta-base" (default: `"roberta-base"`)
        subset (str): The dataset subset, e.g., "trex" (default: `"trex"`)
        dump_file (str): The dump file (default: `"dump"`)

    Raises:
        ValueError: If the `dump_file` already exists
    """
    if os.path.exists(dump_file):
        raise ValueError(f"File exists: {dump_file}")

    if dataset == "lama":
        extract_lama(model, subset, dump_file)
    elif dataset == "squad":
        extract_squad(model, dump_file)
    elif dataset == "honest":
        extract_honest(model, subset, dump_file)
    elif dataset == "crows":
        extract_crows(model, dump_file)
    elif dataset == "stereoset":
        extract_stereoset(model, dump_file)


def extract_lama(model: str = "roberta-base", subset: str = "trex", dump_file: str = "dump"):
    # load mining dataset
    lama_dataset = load_dataset("lama", subset)["train"]
    dataset_size = len(lama_dataset)

    K = 100
    config = copy.deepcopy(lama_baert_mining_config)
    config.update({"K":K,
                   "indexes": list(range(dataset_size))})

    if "bert" in model or "bart" in model:
        config = copy.deepcopy(lama_baert_mining_config)
        config.update({"K": K,
                       "indexes": list(range(dataset_size))})

        if model.startswith("roberta"):
            miner = BaertMiner(model, "roberta", device=device)
        elif model.startswith("microsoft/deberta"):
            miner = BaertMiner(model, "deberta", device=device)
            config = copy.deepcopy(lama_deberta_mining_config)
            config.update({"K": K,
                           "indexes": list(range(dataset_size))})
        elif model.startswith("facebook/bart"):
            miner = BaertMiner(model, "bart", device=device)
    elif "t5" in model:
        miner = T5Miner(model, device=device)
        config = copy.deepcopy(lama_t5_mining_config)
        config.update({"K": K,
                       "indexes": list(range(dataset_size))})
    
    elif "bloom" in model:
        miner = BLOOMMiner(model, device=device)
        config = copy.deepcopy(lama_bloom_mining_config)
        config.update({"K": K,
                       "indexes": list(range(dataset_size))})

    elif "rag" in model:
        miner = RagMiner(model, device=device)
        config = copy.deepcopy(lama_rag_mining_config)
        config.update({"K": K,
                       "indexes": list(range(dataset_size))})

    predictions = miner.mine(lama_dataset, config)

    with open(dump_file, 'a') as log:
        for prediction in predictions:
            log.write(json.dumps(prediction) + "\n")
  
          
def extract_squad(model: str = "roberta-base", dump_file: str = "dump"):
    # load mining dataset
    squad_dataset = load_dataset("lama", "squad")["train"]
    dataset_size = len(squad_dataset)

    K = 100
    config = copy.deepcopy(squad_baert_mining_config)
    config.update({"K":K,
                   "indexes": list(range(dataset_size))})

    if "bert" in model or "bart" in model:
        config = copy.deepcopy(squad_baert_mining_config)
        config.update({"K": K,
                       "indexes": list(range(dataset_size))})

        if model.startswith("roberta"):
            miner = BaertMiner(model, "roberta", device=device)
        elif model.startswith("microsoft/deberta"):
            miner = BaertMiner(model, "deberta", device=device)
            config = copy.deepcopy(squad_deberta_mining_config)
            config.update({"K": K,
                           "indexes": list(range(dataset_size))})
        elif model.startswith("facebook/bart"):
            miner = BaertMiner(model, "bart", device=device)
    elif "t5" in model:
        miner = T5Miner(model, device=device)
        config = copy.deepcopy(squad_t5_mining_config)
        config.update({"K": K,
                       "indexes": list(range(dataset_size))})
    elif "rag" in model:
        miner = RagMiner(model, device=device)
        config = copy.deepcopy(squad_rag_mining_config)
        config.update({"K": K,
                       "indexes": list(range(dataset_size))})

    predictions = miner.mine(squad_dataset, config)

    with open(dump_file, 'a') as log:
        for prediction in predictions:
            log.write(json.dumps(prediction) + "\n")


def extract_honest(model: str = "roberta-base", subset: str = "trex", dump_file: str = "dump"):
    # load mining dataset
    honest_dataset = load_dataset("MilaNLProc/honest", subset)["honest"]
    dataset_size = len(honest_dataset)

    K = 100
    config = copy.deepcopy(honest_baert_mining_config)
    config.update({"K":K,
                   "indexes": list(range(dataset_size))})

    if "bert" in model or "bart" in model:
        config = copy.deepcopy(honest_baert_mining_config)
        config.update({"K": K,
                       "indexes": list(range(dataset_size))})

        if model.startswith("roberta") or model.startswith("distilroberta"):
            miner = BaertMiner(model, "roberta", device=device)
        elif model.startswith("microsoft/deberta"):
            miner = BaertMiner(model, "deberta", device=device)
            config = copy.deepcopy(honest_deberta_mining_config)
            config.update({"K": K,
                           "indexes": list(range(dataset_size))})
        elif model.startswith("facebook/bart") or model.startswith("lucadiliello/bart-small"):
            miner = BaertMiner(model, "bart", device=device)
        elif model.startswith("bert") or model.startswith("distilbert"):
            config.update({"mask_template": "[MASK]"})
            miner = BaertMiner(model, "bert", device=device)
        elif model.startswith("albert"):
            config.update({"mask_template": "[MASK]"})
            miner = BaertMiner(model, "albert", device=device)

    elif "t5" in model:
        miner = T5Miner(model, device=device)
        config = copy.deepcopy(honest_t5_mining_config)
        config.update({"K": K,
                       "indexes": list(range(dataset_size))})
    
    elif "bloom" in model:
        miner = BLOOMMiner(model, device=device)
        config = copy.deepcopy(lama_bloom_mining_config)
        config.update({"K": K,
                       "indexes": list(range(dataset_size))})

    elif "rag" in model:
        miner = RagMiner(model, device=device)
        config = copy.deepcopy(honest_rag_mining_config)
        config.update({"K": K,
                       "indexes": list(range(dataset_size))})

    elif model.startswith("gpt2"):
            miner = GPT2Miner(model, "gpt2", device=device)

    predictions = miner.mine(honest_dataset, config)

    with open(dump_file, 'a') as log:
        for prediction in predictions:
            log.write(json.dumps(prediction) + "\n")


def extract_crows(model: str = "roberta-base", dump_file: str = "dump"):
    # load mining dataset
    dataset = CrowsDataLoader()
    dataset_size = len(dataset)

    K = 100
    config = copy.deepcopy(crows_baert_mining_config)
    config.update({"K":K,
                   "indexes": list(range(dataset_size))})

    if "bert" in model or "bart" in model:
        config = copy.deepcopy(crows_baert_mining_config)
        config.update({"K": K,
                       "indexes": list(range(dataset_size))})

        if model.startswith("roberta"):
            miner = BaertMiner(model, "roberta", device=device)
        elif model.startswith("microsoft/deberta"):
            miner = BaertMiner(model, "deberta", device=device)
            config = copy.deepcopy(crows_deberta_mining_config)
            config.update({"K": K,
                           "indexes": list(range(dataset_size))})
        elif model.startswith("facebook/bart"):
            miner = BaertMiner(model, "bart", device=device)
    elif "t5" in model:
        miner = T5Miner(model, device=device)
        config = copy.deepcopy(crows_t5_mining_config)
        config.update({"K": K,
                       "indexes": list(range(dataset_size))})
    elif "rag" in model:
        miner = RagMiner(model, device=device)
        config = copy.deepcopy(crows_rag_mining_config)
        config.update({"K": K,
                       "indexes": list(range(dataset_size))})

    predictions = miner.mine(dataset, config)

    with open(dump_file, 'a') as log:
        for prediction in predictions:
            log.write(json.dumps(prediction) + "\n")


def extract_stereoset(model: str = "roberta-base", dump_file: str = "dump"):
    # load mining dataset
    dataset = StereosetDataLoader()
    dataset_size = len(dataset)

    K = 100
    config = copy.deepcopy(stereoset_baert_mining_config)
    config.update({"K":K,
                   "indexes": list(range(dataset_size))})

    if "bert" in model or "bart" in model:
        config = copy.deepcopy(stereoset_baert_mining_config)
        config.update({"K": K,
                       "indexes": list(range(dataset_size))})

        if model.startswith("roberta"):
            miner = BaertMiner(model, "roberta", device=device)
        elif model.startswith("microsoft/deberta"):
            miner = BaertMiner(model, "deberta", device=device)
            config = copy.deepcopy(stereoset_deberta_mining_config)
            config.update({"K": K,
                           "indexes": list(range(dataset_size))})
        elif model.startswith("facebook/bart"):
            miner = BaertMiner(model, "bart", device=device)
    elif "t5" in model:
        miner = T5Miner(model, device=device)
        config = copy.deepcopy(stereoset_t5_mining_config)
        config.update({"K": K,
                       "indexes": list(range(dataset_size))})
    elif "rag" in model:
        miner = RagMiner(model, device=device)
        config = copy.deepcopy(stereoset_rag_mining_config)
        config.update({"K": K,
                       "indexes": list(range(dataset_size))})

    predictions = miner.mine(dataset, config)

    with open(dump_file, 'a') as log:
        for prediction in predictions:
            log.write(json.dumps(prediction) + "\n")


if __name__ == '__main__':
    fire.Fire(extract)
