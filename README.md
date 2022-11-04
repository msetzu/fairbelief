# Quickstart
## Install dependencies
```shell
mkvirtualenv -p python3.9 fairbelief
pip install -r src/requirements.txt
```

## Extracting beliefs
You can extract beliefs from any of the supported model families, currently `T5`, `Bert`, `RoBERTa`, and `DeBERTa`.
Prompts should follow
Here is an example on `roberta-base`.

```python
from datasets import load_dataset
import numpy

from miners.baert import BaertMiner
from miners.mine import LAMA_BAERT_MINER

# load mining dataset
lama_dataset = load_dataset("lama")
dataset_size = len(lama_dataset)

# mining
nr_random_entries = 1000
random_entries_indexes = numpy.random.randint(low=0, high=dataset_size,
                                              size=nr_random_entries)
mining_cfg = LAMA_BAERT_MINER(K=100, indexes=random_entries_indexes)
miner = BaertMiner("roberta-base", "roberta", device="cuda")
predictions = miner.mine(lama_dataset["train"],
                         mining_cfg)
```

## Notebooks
You can find an example usage in the `notebooks/mining.ipynb` Jupyter Notebook.

---

# Apply on your own dataset
## Miners
You can find miners for some popular models in `miners`, to apply to your own dataset implement the `miners.mine.Miner` interface:
```python
class Miner:
    """Miner class, extracts beliefs from the provided models."""
    def __init__(self):
        pass

    @abstractmethod
    def mine(self, prompts: DataLoader, config: MinerConfig) -> List:
        pass
```

## Mining configurations
To apply it to your prompt `DataLoader` you need to instantiate a `MinerConfig`:
```python
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
```
We provide some predefined `MinerConfig` for some prompting datasets, like `LAMA`.
You can find them in `miners.mine`:
```python
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
    mask_template="<extra_id0>",
    fillin_column="obj_surface",
    triples=False
)

LAMA_T5_TRIPLES_MINER = MinerConfig(
    template_column="template",
    mask_template="<extra_id0>",
    fillin_column="obj_surface",
    subject_template="[X]",
    subject_value="sub_surface",
    object_template="[Y]",
    triples=True
)
```
















