import itertools
import os
import random
import sys

__SRC = os.path.abspath(".") + "/.."
__DATA = __SRC + "/../data"
__NOTEBOOKS = __SRC + "notebooks"

sys.path.append(__SRC)


import pandas
from datasets import load_dataset

from preprocessing.stereoset import StereosetDataLoader
from preprocessing.crows import CrowsDataLoader


import fire

from miners.mine import Miner

random.seed(42)


def relations_for(data, beliefs, dataset, subset) -> list:
    """Compute relations for the given `beliefs_file`."""
    if dataset == "lama":
        if subset == "trex":
            target_mask = "[MASK]"
            relation_column = "predicate_id"
            template_column = "masked_sentence"
    elif dataset == "honest":
        target_mask = "[M]"
        template_column = "template_masked"
        relation_column = "type"
    elif dataset == "stereoset":
        target_mask = "[MASK]"
        template_column = "template_masked"
        relation_column = "bias_type"
    elif dataset == "crows":
        target_mask = "[MASK]"
        template_column = "template_masked"
        relation_column = "bias_type"
    else:
        raise ValueError(f"Unknown dataset, subset: {dataset, subset}")

    templates = (result["input_query"] for result in beliefs)
    results_ids = (template.replace("[MASK]", target_mask)
                   .replace("<mask>", target_mask)
                   .replace("<extra_id0>", target_mask)
                   .replace("<extra_id_0>", target_mask)
                   for template in templates)
    if dataset == "crows":
        relations = [data[i]["bias_type"] for i in range(len(data))]
    elif dataset == "stereoset":
        relations = list(itertools.chain.from_iterable([[data.huggingface_dataset[i]["bias_type"]] *
                                                        (len(data.huggingface_dataset[i]["sentences"]["sentence"]) - 1)
                                                        for i in range(len(data.huggingface_dataset))]))
    else:
        relations = list()
        for result_id in results_ids:
            relations.append(data.filter(lambda x: x[template_column] == result_id)[0][relation_column])

    return relations


def extract(dataset, subset, top_p, k):
    beliefs_files = os.listdir(f"{__DATA}/beliefs/")
    beliefs_files = [f for f in beliefs_files if f.startswith(f"{dataset}_{subset}")
                     and "t5_" not in f and "rag_" not in f]
    beliefs_models = list(itertools.chain([f.split("_")[-1].split(".jsonl")[0]
                                           for f in beliefs_files]))
    beliefs = [Miner.load_mining_results(f"{__DATA}/beliefs/{f}") for f in beliefs_files]

    if dataset == "lama":
        data = load_dataset("lama", subset)["train"]
    elif dataset == "honest":
        data = load_dataset("MilaNLProc/honest", subset)["honest"]
    elif dataset == "squad":
        data = load_dataset("lama", "squad")["train"]
    elif dataset == "crows":
        data = CrowsDataLoader()
    elif dataset == "stereoset":
        data = StereosetDataLoader()

    relations_per_model = [relations_for(data, b, dataset, subset) for b in beliefs]
    if dataset in ["stereoset", "crows"]:
        sorted_relations_per_model = [sorted(enumerate(rels), key=lambda x: x[1]) for rels in relations_per_model]
        groups_per_model = [itertools.groupby(relations, key=lambda x: x[1]) for relations in sorted_relations_per_model]
    else:
        groups_per_model = [itertools.groupby(enumerate(relations), key=lambda x: x[1]) for relations in relations_per_model]
    groups_per_model = [{relation: [i for i, _ in group] for relation, group in groups} for groups in groups_per_model]
    nr_relations = len(groups_per_model[0])
    K = k * nr_relations

    # every model has the same queries, so we can use any of them
    relation_samples = {relation: random.sample(group, k=min(K, len(group))) for _ in range(nr_relations)
                        for relation, group in groups_per_model[0].items()}

    tables = list()
    for model_beliefs, model in zip(beliefs, beliefs_models):
        selected_samples = dict()
        for relation, samples_indices in relation_samples.items():
            relation_beliefs = [belief for belief in model_beliefs if belief["index"] in samples_indices]
            relation_beliefs = [{"index": belief["index"],
                                "input_query": belief["input_query"],
                                 "predictions": belief["predictions"][:top_p]}
                                for belief in relation_beliefs]
            selected_samples[relation] = [relation_beliefs[i * k: (i + 1) * k] for i in range(nr_relations)]

        table = list()
        for relation in selected_samples:
            for sample_index, samples in enumerate(selected_samples[relation]):
                for belief in samples:
                    for rank, prediction in enumerate(belief["predictions"]):
                        table.append((sample_index, belief["index"], relation,
                                      belief["input_query"], rank, prediction))

        table = pandas.DataFrame(table, columns=["sample_index", "belief_index", "relation",
                                                 "template", "p", f"{model}"])
        tables.append(table)
        # table.to_csv(output, index=False, header=False)
    full_df = pandas.concat(tables, axis="columns")
    full_df["dataset"] = dataset
    full_df["subset"] = subset
    full_df = full_df.loc[:, ~full_df.columns.duplicated()].copy()
    assignments = {0: "marta", 1: "mattia", 2: "pasquale"}
    full_df.sample_index = full_df.sample_index.apply(lambda x: assignments[int(x) % 3])
    print(f"Dumping to {os.getcwd()}/{dataset}_{subset}_qualitative.csv")
    full_df.to_csv(f"{dataset}_{subset}_qualitative.csv", index=False)


if __name__ == '__main__':
    fire.Fire(extract)
