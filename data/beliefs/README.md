# Beliefs
Beliefs extracted from each model on the LAMA benchmark.
File `{model}_lama.jsonl` is a JSON-lines file where each line contains the `ground_truth` prediction on LAMA, the `predictions` of the model `model`, e.g., `roberta_lama.jsonl` contains the predictions of RoBERTa on LAMA.
Additionally, we also extracted beliefs on the triple-like version of LAMA, and stored such beliefs in files `{model}_lama_on_triples.jsonl`.
Beliefs of retrieval-augmented models also contain a `documents` entry where the retrieved documents are contained.