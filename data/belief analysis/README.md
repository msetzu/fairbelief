# Beliefs analysis
Precision@K of the beliefs extracted from each model on the LAMA benchmark.
File `{model}_precisions.npy` is a numpy file holding both `K` and the respective precision.
Additionally, the same analysis has been run on beliefs extracted from the triple-like version of LAMA, and stored such beliefs in files `{model}_precisions_on_triples.npy`.
Finally, we store precisions by relationship ID in files `{model}_precisions_on_{ID}.npy`.