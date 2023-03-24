from datasets import load_dataset

from miners.mine import Miner


def relations_for(beliefs_file, dataset, subset) -> list:
    """Compute relations for the given `beliefs_file`."""
    dataset = load_dataset(dataset, subset)
    results = Miner().load_mining_results(beliefs_file)

    if dataset == "lama":
        if subset == "trex":
            dataset = dataset["train"]
            target_mask = "[MASK]"
            relation_column = "predicate_id"
            template_column = "masked_sentence"
    elif dataset == "MilaNLProc/honest":
        dataset = dataset["honest"]
        target_mask = "[M]"
        template_column = "template_masked"
        relation_column = "type"
    elif dataset == "stereoset":
        dataset = dataset["intrasentence"]
        target_mask = "[MASK]"
        template_column = "template_masked"
        relation_column = "bias_type"
    elif dataset == "crows":
        target_mask = "[MASK]"
        template_column = "sent_more"
        relation_column = "bias_type"
    else:
        raise ValueError(f"Unknown dataset, subset: {dataset, subset}")

    templates = (result["input_query"] for result in results)
    results_ids = (template.replace("[MASK]", target_mask)
                   .replace("<mask>", target_mask)
                   .replace("<extra_id0>", target_mask)
                   .replace("<extra_id_0>", target_mask)
                   for template in templates)
    if dataset == "crows":
        relations = [dataset[dataset[template_column].startsWith(result_id[:result_id.index(target_mask)])]
                     for result_id in results_ids]
    else:
        relations = [dataset[dataset[template_column] == result_id][relation_column] for result_id in results_ids]

    return relations
