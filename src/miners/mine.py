import json
from abc import abstractmethod
from typing import List, Dict, Tuple, Optional

import numpy
from torch.utils.data import DataLoader


class MinerConfig(dict):
    """
    Attributes:
        template_column: The template column where the prompt is
        subject_template: If `triples == True` then the template for the `subject_value`
        subject_value: If `triples == True` then the value to replace to `subject_template`
        object_template: If `triples == True` then the template for the mask
        mask_template: Template to replace the "[MASK]" in the template
        fillin_column: Ground truth column holding the fill-in for the mask
        triples: True if mining from triples
        K: Top-K predictions to mine
        indexes: Instances to mine, by index
    """
    def __init__(self, template_column: str="template", subject_template: str="[X]",
                 subject_value: str="sub_surface", object_template: str="[Y]",
                 original_mask: str="[MASK]", mask_template: str="<mask>", fillin_column: str= "obj_surface",
                 triples: bool = False,
                 K: int = 10, indexes: Optional[numpy.ndarray] = None):
        """
        Configuration for mining.
        Args:
            triples: True if the data is presented as triples, False otherwise. Defaults to False
            template_column: Name of the template column if `triples == True`. Defaults to "template"
            subject_template: Subject template column if `triples == True`. Defaults to "[X]"
            object_template: Object template column if `triples == True`. Defaults to "[Y]"
            subject_value: Subject value column if `triples == True`. Defaults to "sub_surface"
            mask_template: Mask token to use. Defaults to "<mask>"
            fillin_column: Ground truth column holding the fill-in for the mask. Defaults to "obj_surface"
            K: Top-K predictions to mine. Defaults to 10
            indexes: Only mine the dataset on these indexes, `None` to mine on all. Defaults to None
        """
        super().__init__()
        self["template_column"] = template_column
        self["subject_template"] = subject_template
        self["subject_value"] = subject_value
        self["object_template"] = object_template
        self["mask_template"] = mask_template
        self["original_mask"] = original_mask
        self["fillin_value"] = fillin_column
        self["triples"] = triples
        self["K"] = K
        self["indexes"] = indexes


class Miner:
    """Miner class, extracts beliefs from the provided models."""
    def __init__(self):
        pass

    @abstractmethod
    def mine(self, prompts: DataLoader, config: MinerConfig) -> List:
        pass

    def load_predictions(self, predictions_file: str) -> Tuple[List, List]:
        ground_truths = list()
        predictions = list()
        with open(predictions_file, 'r') as log:
            for instance in log:
                ground_truths.append(instance[3])
                predictions.append(instance[4])

        return ground_truths, predictions

    @staticmethod
    def load_mining_results(predictions_file: str) -> List[Dict]:
        results = list()
        with open(predictions_file, 'r') as log:
            for instance in log:
                try:
                    index, uuid, input, predictions, ground_truth = json.loads(instance)
                    results.append({
                        "index": index,
                        "uuid": uuid,
                        "input_query": input,
                        "predictions": predictions,
                        "ground_truth": ground_truth
                    })
                except ValueError:
                    index, uuid, input, predictions, ground_truth, documents = json.loads(instance)
                    results.append({
                        "index": index,
                        "uuid": uuid,
                        "documents": documents,
                        "input_query": input,
                        "predictions": predictions,
                        "ground_truth": ground_truth
                    })

        return results

### Stock configurations
## LAMA
# Baert
LAMA_BAERT_MINER = MinerConfig(
    template_column="masked_sentence",
    mask_template="<mask>",
    fillin_column="obj_surface",
    triples=False
)

LAMA_BAERT_TRIPLES_MINER = MinerConfig(
    template_column="template",
    mask_template="<mask>",
    fillin_column="obj_surface",
    subject_template="[X]",
    subject_value="sub_surface",
    object_template="[Y]",
    triples=True
)

LAMA_DEBERTA_MINER = MinerConfig(
    template_column="masked_sentence",
    mask_template="[MASK]",
    fillin_column="obj_surface",
    triples=False
)

LAMA_DEBERTA_TRIPLES_MINER = MinerConfig(
    template_column="template",
    mask_template="[MASK]",
    fillin_column="obj_surface",
    subject_template="[X]",
    subject_value="sub_surface",
    object_template="[Y]",
    triples=True
)

# RAG
LAMA_RAG_MINER = MinerConfig(
    template_column="masked_sentence",
    mask_template="<mask>",
    fillin_column="obj_surface",
    triples=False
)

LAMA_RAG_TRIPLES_MINER = MinerConfig(
    template_column="template",
    mask_template="<mask>",
    fillin_column="obj_surface",
    subject_template="[X]",
    subject_value="sub_surface",
    object_template="[Y]",
    triples=True
)

# T5
LAMA_T5_MINER = MinerConfig(
    template_column="masked_sentence",
    mask_template="<extra_id_0>",
    fillin_column="obj_surface",
    triples=False
)

LAMA_BLOOM_MINER = MinerConfig(
    template_column="masked_sentence",
    mask_template="<extra_id_0>",
    fillin_column="obj_surface",
    triples=False
)

LAMA_OPT_MINER = MinerConfig(
    template_column="masked_sentence",
    mask_template="<extra_id_0>",
    fillin_column="obj_surface",
    triples=False
)

LAMA_T5_TRIPLES_MINER = MinerConfig(
    template_column="template",
    mask_template="<extra_id_0>",
    fillin_column="obj_surface",
    subject_template="[X]",
    subject_value="sub_surface",
    object_template="[Y]",
    triples=True
)

# Baert
SQUAD_BAERT_MINER = MinerConfig(
    template_column="masked_sentence",
    mask_template="<mask>",
    fillin_column="obj_label",
    triples=False
)

SQUAD_BAERT_TRIPLES_MINER = MinerConfig(
    template_column="template",
    mask_template="<mask>",
    fillin_column="obj_label",
    subject_template="[X]",
    subject_value="sub_surface",
    object_template="[Y]",
    triples=True
)

SQUAD_DEBERTA_MINER = MinerConfig(
    template_column="masked_sentence",
    mask_template="[MASK]",
    fillin_column="obj_label",
    triples=False
)

SQUAD_DEBERTA_TRIPLES_MINER = MinerConfig(
    template_column="template",
    mask_template="[MASK]",
    fillin_column="obj_label",
    subject_template="[X]",
    subject_value="sub_surface",
    object_template="[Y]",
    triples=True
)

# RAG
SQUAD_RAG_MINER = MinerConfig(
    template_column="masked_sentence",
    mask_template="<mask>",
    fillin_column="obj_label",
    triples=False
)

SQUAD_RAG_TRIPLES_MINER = MinerConfig(
    template_column="template",
    mask_template="<mask>",
    fillin_column="obj_label",
    subject_template="[X]",
    subject_value="sub_surface",
    object_template="[Y]",
    triples=True
)

# T5
SQUAD_T5_MINER = MinerConfig(
    template_column="masked_sentence",
    mask_template="<extra_id_0>",
    fillin_column="obj_label",
    triples=False
)

SQUAD_T5_TRIPLES_MINER = MinerConfig(
    template_column="template",
    mask_template="<extra_id_0>",
    fillin_column="obj_label",
    subject_template="[X]",
    subject_value="sub_surface",
    object_template="[Y]",
    triples=True
)

## HONEST
# Baert
HONEST_BAERT_MINER = MinerConfig(
    template_column="template_masked",
    mask_template="<mask>",
    original_mask="[M]",
    fillin_column="type",
    triples=False
)

HONEST_DEBERTA_MINER = MinerConfig(
    template_column="template_masked",
    mask_template="[MASK]",
    original_mask="[M]",
    fillin_column="type",
    triples=False
)

# T5
HONEST_T5_MINER = MinerConfig(
    template_column="template_masked",
    mask_template="<extra_id_0>",
    original_mask="[M]",
    fillin_column="type",
    triples=False
)

HONEST_BLOOM_MINER = MinerConfig(
    template_column="template_masked",
    mask_template="[MASK]",
    original_mask="[M]",
    fillin_column="type",
    triples=False
)

HONEST_OPT_MINER = MinerConfig(
    template_column="template_masked",
    mask_template="[MASK]",
    original_mask="[M]",
    fillin_column="type",
    triples=False
)

# RAG
HONEST_RAG_MINER = MinerConfig(
    template_column="template_masked",
    mask_template="<mask>",
    original_mask="[M]",
    fillin_column="type",
    triples=False
)

# Realm
HONEST_REALM_MINER = MinerConfig(
    template_column="template_masked",
    mask_template="<mask>",
    original_mask="[M]",
    fillin_column="type",
    triples=False
)

## Stereoset
STEREOSET_BAERT_MINER = MinerConfig(
    template_column="template_masked",
    mask_template="<mask>",
    original_mask="[MASK]",
    fillin_column="obj_surface",
    triples=False
)

STEREOSET_DEBERTA_MINER = MinerConfig(
    template_column="template_masked",
    mask_template="[MASK]",
    original_mask="[MASK]",
    fillin_column="obj_surface",
    triples=False
)

STEREOSET_T5_MINER = MinerConfig(
    template_column="template_masked",
    mask_template="<extra_id_0>",
    original_mask="[MASK]",
    fillin_column="obj_surface",
    triples=False
)

STEREOSET_RAG_MINER = MinerConfig(
    template_column="template_masked",
    mask_template="<mask>",
    original_mask="[MASK]",
    fillin_column="obj_surface",
    triples=False
)


## Crows
CROWS_BAERT_MINER = MinerConfig(
    template_column="template_masked",
    mask_template="<mask>",
    original_mask="[MASK]",
    fillin_column="obj_surface",
    triples=False
)

CROWS_DEBERTA_MINER = MinerConfig(
    template_column="template_masked",
    mask_template="[MASK]",
    original_mask="[MASK]",
    fillin_column="obj_surface",
    triples=False
)

CROWS_RAG_MINER = MinerConfig(
    template_column="template_masked",
    mask_template="<mask>",
    original_mask="[MASK]",
    fillin_column="obj_surface",
    triples=False
)

CROWS_REALM_MINER = MinerConfig(
    template_column="template_masked",
    mask_template="<mask>",
    original_mask="[MASK]",
    fillin_column="obj_surface",
    triples=False
)

CROWS_T5_MINER = MinerConfig(
    template_column="template_masked",
    mask_template="<extra_id_0>",
    original_mask="[MASK]",
    fillin_column="obj_surface",
    triples=False
)
