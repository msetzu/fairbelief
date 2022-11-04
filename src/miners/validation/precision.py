from typing import Iterable

from sentence_transformers import util
from transformers import AutoModel
import numpy


def precision_at(predicted: Iterable, ground_truth: Iterable, k: int = 1) -> float:
    """
    Compute precision@k for the given `predictions` and `ground_truths`
    Args:
        predicted: The model predictions
        ground_truth: Ground truth predictions
        k: The value of k

    Returns:
        The precision@k of the given `predictions`
    """
    return sum([any([p_i.lower() == g.lower() for p_i in p[:k]]) for p, g in zip(predicted, ground_truth)]) / len(predicted)


def precisions_at(predicted: Iterable, ground_truth: Iterable, K: int = 10) -> numpy.ndarray:
    """
    Compute precision@k for the given `predictions` and `ground_truths` with k in [1, 2,  .. K]
    Args:
        predicted: The model predictions
        ground_truth: Ground truth predictions
        K: The maximum value of k. Defaults to 10

    Returns:
        The precision@k of the given `predictions`, for k = [1, 2, .. K]
    """
    precisions = numpy.array([precision_at(predicted, ground_truth, k=k) for k in range(K)])
    precisions = numpy.vstack((numpy.arange(1, K), precisions))

    return precisions


def cosine_similarity(predicted: Iterable, ground_truth: Iterable, similarity_model: str = "sentence-transformers/all-MiniLM-L6-v2") -> numpy.ndarray:
    """
    Compute cosine similarity between the predicted values and the ground truth
    Args:
        predicted: Predicted values
        ground_truth: Ground truths
        similarity_model: The similarity embedding to use. Defaults to "sentence-transformers/all-MiniLM-L6-v2"

    Returns:
        A cosine similarity matrix where entry i, j holds the cosine similarity between the i-th ground truth and the j-th
        prediction of the model on that instance
    """
    model = AutoModel.from_pretrained(similarity_model, load_in_8bit=True)

    return numpy.array([[util.pytorch_cos_sim(model.encode(p, convert_to_tensor=True),
                                              model.encode(g, convert_to_tensor=True)).item()
                         for p in instance_predictions]
                        for instance_predictions, g in zip(predicted, ground_truth)])
