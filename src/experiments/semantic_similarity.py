import os
import sys

__SRC = os.path.abspath(".") + "/.."
__DATA = __SRC + "/../data"
__DATA = "/home/mattiasetzu/disk1/data"
__DATA = "./"
__NOTEBOOKS = __SRC + "notebooks"

sys.path.append(__SRC)

from sentence_transformers import SentenceTransformer
import pandas, numpy
import fire
import torch
from scipy.spatial.distance import pdist

from tqdm import tqdm
from alive_progress import alive_bar

from miners.mine import Miner


device = "cuda" if torch.cuda.is_available() else "cpu"
similarity_model = SentenceTransformer("all-MiniLM-L6-v2").to(device)


def similarity_at(models_beliefs, K, single=False):
    n_beliefs = len(models_beliefs[0])
    predictions = numpy.array([[[[similarity_model.encode(model_beliefs[template]["input_query"].replace("[mask]", "").replace("[MASK]", "").replace("[M]", "").replace("[m]", "").replace("<mask>", "").replace("<MASK>", "")\
                                  + " " + model_beliefs[template]["predictions"][k])]
                                  for model_beliefs in models_beliefs]
                                for template in range(n_beliefs)]
                              for k in tqdm(range(K), total=K)]).squeeze()
    similarities = list()
    if not single:
        for k in range(K):
            similarities_at_k = list()
            for t in range(predictions.shape[1]):
                encodings_point_k = numpy.vstack([predictions[current_k, t, :, :].squeeze()
                                                  for current_k in range(k + 1)])
                similarity_at_k_and_t = pdist(encodings_point_k, metric="cosine").mean()
                similarities_at_k.append(similarity_at_k_and_t)
            similarities_at_k = numpy.array(similarities_at_k)
            similarities.append(similarities_at_k)
    else:
        for k in range(K):
            similarities_at_k = list()
            for t in range(predictions.shape[1]):
                # need the -1 to get the 0-th prediction
                encodings_point_k = predictions[k, t, :, :].squeeze()
                similarity_at_k_and_t = pdist(encodings_point_k, metric="cosine").mean()
                similarities_at_k.append(similarity_at_k_and_t)
            similarities_at_k = numpy.array(similarities_at_k)
            similarities.append(similarities_at_k)
    similarities = numpy.array(similarities)
    
    return similarities


def by_family(subset, K, single=False):
    families = ["roberta", "bart", "bert", "albert"]
    beliefs_files = os.listdir(f"{__DATA}/beliefs/")
    beliefs_files = [f for f in beliefs_files if f.startswith(f"honest_{subset}") and f.endswith(".jsonl")]
    beliefs_files = sorted(beliefs_files)
    beliefs_files_by_family = [[b for b in beliefs_files if b.startswith(f"honest_{subset}_{family}")] for family in families]
    beliefs_by_family = [[Miner.load_mining_results(f"{__DATA}/beliefs/{f}") for f in family_beliefs]
                          for family_beliefs in beliefs_files_by_family]

    table = list()
    n_beliefs = len(beliefs_by_family[0][0])
    for family_beliefs, family_name in zip(beliefs_by_family, families):
        print(family_name)
        family_similarity_at = similarity_at(family_beliefs, K, single)
        for k in range(K):
            for t in range(n_beliefs):
                table.append([family_name, subset, k + 1, t, family_similarity_at[k, t].item()])
        full_df = pandas.DataFrame(table)
        full_df.columns = ["family", "subset", "k", "template", "similarity_at_k"]
        if not single:
            full_df.to_csv(f"honest_{subset}.similarity.by_family.cumulative.csv", index=False)
        else:
            full_df.to_csv(f"honest_{subset}.similarity.by_family.point_k.csv", index=False)
        print("\tdone.\n")


def by_scale(subset, K, single=False):
    beliefs_files = os.listdir(f"{__DATA}/beliefs/")
    beliefs_files = [f for f in beliefs_files if f.startswith(f"honest_{subset}") and f.endswith(".jsonl")]
    beliefs_files = sorted(beliefs_files)
    beliefs = [Miner.load_mining_results(f"{__DATA}/beliefs/{f}") for f in beliefs_files]

    table = list()
    n_beliefs = len(beliefs[0])
    scale_similarity_at = similarity_at(beliefs, K, single)
    for k in range(K):
        for t in range(n_beliefs):
            table.append([subset, k + 1, t, scale_similarity_at[k, t].item()])
        full_df = pandas.DataFrame(table)
        full_df.columns = ["subset", "k", "template", "similarity_point_k"]
        if not single:
            full_df.to_csv(f"honest_{subset}.similarity.by_scale.cumulative.csv", index=False)
        else:
            full_df.to_csv(f"honest_{subset}.similarity.by_scale.point_k.csv", index=False)

def by_model_scale(subset, K, single=False):
    beliefs_files = os.listdir(f"{__DATA}/beliefs/")
    beliefs_files = [f for f in beliefs_files if f.startswith(f"honest_{subset}") and f.endswith(".jsonl")]
    beliefs_files = sorted(beliefs_files)
    small_beliefs_files = [b for b in beliefs_files if "small.jsonl" in b]
    large_beliefs_files = [b for b in beliefs_files if "large.jsonl" in b]
    medium_beliefs_files = [b for b in beliefs_files if b not in small_beliefs_files + large_beliefs_files]
    beliefs = [Miner.load_mining_results(f"{__DATA}/beliefs/{f}") for f in beliefs_files]

    table = list()
    n_beliefs = len(beliefs[0])
    beliefs_by_scale = [small_beliefs_files, medium_beliefs_files, large_beliefs_files]
    for scale, beliefs_by_scale in zip(("small", "medium", "large"), beliefs_by_scale):
        scale_similarity_at = similarity_at(beliefs, K, single)
        for k in range(K):
            for t in range(n_beliefs):
                table.append([subset, k + 1, t, scale, scale_similarity_at[k, t].item()])
            full_df = pandas.DataFrame(table)
            # full_df.columns = ["subset", "k", "template", "scale", "similarity_point_k"]
            if not single:
                full_df.to_csv(f"honest_{subset}.similarity.by_model_scale.cumulative.csv", index=False)
            else:
                full_df.to_csv(f"honest_{subset}.similarity.by_model_scale.point_k.csv", index=False)


def by_family_and_identity(subset, K, single=False):
    families = ["roberta", "bart", "bert", "albert"]
    binary_identities = ["girlfriend", "he", "she", "the aunt", "the boy",  "the boyfriend",  "the bride",  "the brother",  "the child",  "the children",  "the daughter",  "the father",  "the female",  "the gentleman",  "the girl",  "the girlfriends",  "the groom",  "the husband",  "the ladies",  "the lady",  "the lords",  "the male",  "the man",  "the men",  "the mother",  "the sister",  "the son",  "the spouses",  "the uncle",  "the wife",  "the wives",  "the woman",  "the women",  "the young"]
    queer_identities = ["Fae",  "LGBT person",  "LGBTQ person",  "LGBTQI person",  "LGBTQIA person",  "LGBTQIA+ person",  "They",  "Xe",  "Ze",  "agender",  "androgynous",  "aromantic",  "asexual",  "biromantic",  "bisexual",  "catgender",  "cisgender",  "demigender",  "demisexual",  "enby",  "gay",  "gender conforming",  "gender non-conforming",  "genderfluid",  "genderless",  "genderqueer",  "heteroromantic",  "heterosexual",  "homoromantic",  "homosexual",  "intersexual",  "lesbian",  "non-binary",  "nonqueer",  "pangender",  "panromantic",  "pansexual",  "polygender",  "queer",  "straight",  "trans",  "transgender",  "transman",  "transsexual",  "transwoman",  "xenogender"]
    identities = binary_identities if subset == "en_binary" else queer_identities
    identities = {h for h in identities if h[:-1] not in identities}
    # identities = [i.replace("the ", "") for i in identities]

    beliefs_files = os.listdir(f"{__DATA}/beliefs/")
    beliefs_files_by_family = [[f for f in beliefs_files if f.startswith(f"honest_{subset}_{family}") and f.endswith(".jsonl")]
                                for family in families]
    beliefs_files_by_family = sorted(beliefs_files_by_family)
    beliefs_by_family = [[Miner.load_mining_results(f"{__DATA}/beliefs/{f}") for f in family_beliefs]
                          for family_beliefs in beliefs_files_by_family]

    table = list()
    with alive_bar(total=len(identities)) as bar:
        for identity in identities:
            print(f"identity: {identity}")
            for family_name, family_beliefs in zip(families, beliefs_by_family):
                beliefs_with_identity = [[b for b in model_beliefs if b["input_query"].lower().startswith(identity.lower() + " ") or\
                                          b["input_query"].lower().startswith("the " + identity.lower() + " ")]
                                          for model_beliefs in family_beliefs]
                if len(beliefs_with_identity[0]) > 0:
                    scale_similarity_at = similarity_at(beliefs_with_identity, K, single)

                    for t in range(len(beliefs_with_identity)):
                        for k in range(K):
                            table.append([subset, k + 1, t, identity, family_name,
                                        scale_similarity_at[k, t].item()])
                    full_df = pandas.DataFrame(table)
                    full_df.columns = ["subset", "k", "template_index", "identity", "family", "similarity_point_k"]
                    if not single:
                        full_df.to_csv(f"honest_{subset}.similarity.by_family_and_identity.cumulative.csv", index=False)
                    else:
                        full_df.to_csv(f"honest_{subset}.similarity.by_family_and_identity.point_k.csv", index=False)

            bar()


def by_scale_and_identity(subset, K, single=False):
    binary_identities = ["girlfriend", "he", "she", "the aunt", "the boy",  "the boyfriend",  "the bride",  "the brother",  "the child",  "the children",  "the daughter",  "the father",  "the female",  "the gentleman",  "the girl",  "the girlfriends",  "the groom",  "the husband",  "the ladies",  "the lady",  "the lords",  "the male",  "the man",  "the men",  "the mother",  "the sister",  "the son",  "the spouses",  "the uncle",  "the wife",  "the wives",  "the woman",  "the women",  "the young"]
    queer_identities = ["Fae",  "LGBT person",  "LGBTQ person",  "LGBTQI person",  "LGBTQIA person",  "LGBTQIA+ person",  "They",  "Xe",  "Ze",  "agender",  "androgynous",  "aromantic",  "asexual",  "biromantic",  "bisexual",  "catgender",  "cisgender",  "demigender",  "demisexual",  "enby",  "gay",  "gender conforming",  "gender non-conforming",  "genderfluid",  "genderless",  "genderqueer",  "heteroromantic",  "heterosexual",  "homoromantic",  "homosexual",  "intersexual",  "lesbian",  "non-binary",  "nonqueer",  "pangender",  "panromantic",  "pansexual",  "polygender",  "queer",  "straight",  "trans",  "transgender",  "transman",  "transsexual",  "transwoman",  "xenogender"]
    identities = binary_identities if subset == "en_binary" else queer_identities
    identities = {h for h in identities if h[:-1] not in identities}
    # identities = [i.replace("the ", "") for i in identities]

    beliefs_files = os.listdir(f"{__DATA}/beliefs/")
    beliefs_files = [f for f in beliefs_files if f.startswith(f"honest_{subset}") and f.endswith(".jsonl")]
    beliefs_files = sorted(beliefs_files)
    beliefs = [Miner.load_mining_results(f"{__DATA}/beliefs/{f}") for f in beliefs_files]

    table = list()
    with alive_bar(total=len(identities)) as bar:
        for identity in identities:
            beliefs_with_identity = [[b for b in model_beliefs if b["input_query"].lower().startswith(identity.lower() + " ")]
                                      for model_beliefs in beliefs]
            if len(beliefs_with_identity[0]) > 0:
                scale_similarity_at = similarity_at(beliefs_with_identity, K, single)
                for k in range(K):
                    for t in range(len(beliefs_with_identity)):
                        table.append([subset, k + 1, t, identity,
                                    scale_similarity_at[k, t].item()])
                full_df = pandas.DataFrame(table)
                full_df.columns = ["subset", "k", "template_index", "identity", "similarity_point_k"]
                if not single:
                    full_df.to_csv(f"honest_{subset}.similarity.by_scale_and_identity.cumulative.csv", index=False)
                else:
                    full_df.to_csv(f"honest_{subset}.similarity.by_scale_and_identity.point_k.csv", index=False)

            bar()


if __name__ == "__main__":
    fire.Fire()
