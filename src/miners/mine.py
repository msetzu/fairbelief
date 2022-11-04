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
    def __init__(self, template_column: str= "template", subject_template: str= "[X]",
                 subject_value: str="sub_surface", object_template: str="[Y]",
                 mask_template: str="<mask>", fillin_column: str= "obj_surface",
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
        self.template_column = template_column
        self.subject_template = subject_template
        self.subject_value = subject_value
        self.object_template = object_template
        self.mask_template = mask_template
        self.fillin_template = fillin_column
        self.triples = triples
        self.K = K
        self.indexes = indexes


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
                instance_dict = json.loads(instance)
                ground_truths.append(instance_dict["ground_truth_prediction"])
                predictions.append(instance_dict["prediction"])

        return ground_truths, predictions


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

# T5
LAMA_T5_MINER = MinerConfig(
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
