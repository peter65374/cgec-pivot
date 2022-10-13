"""
Python 3.6
Required:
sudo pip3 install torch
sudo pip3 install sentence-transformers
sudo pip3 install transformers
sudo pip3 install sklearn
"""

import logging
from typing import Text, Any

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import euclidean_distances

logger = logging.getLogger("SIMILARITY")
model = None


def getSimilarityAndDistance(sentence1, sentence2):
    global model
    if model is None:
        logger.info("START - 加载 SIMILARITY 模型")
        model = SentenceTransformer('distiluse-base-multilingual-cased-v2')
        logger.info("FINISH - 加载 SIMILARITY 模型")

    embedding1 = model.encode(sentence1)
    embedding2 = model.encode(sentence2)
    return getSimilarity(embedding1, embedding2), getDistance(embedding1, embedding2)


def getSimilarity(embedding1, embedding2, method="cos"):
    # type: (Any, Any, Text) -> float
    try:
        tensor_1 = getTensor(embedding1)
        tensor_2 = getTensor(embedding2)

        # get similarity
        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        result = cos(tensor_1, tensor_2)
        similarity = result.item()

        if method == "cos":
            return similarity
        else:
            return similarity
    except Exception as e:
        print("Exception thrown during get distance", e)
        return 0


def getDistance(embedding1, embedding2, method="euclidean"):
    # type: (Any, Any, Text) -> float
    try:
        tensor_1 = getTensor(embedding1)
        tensor_2 = getTensor(embedding2)

        # get e distance
        result = euclidean_distances([tensor_1.numpy()], [tensor_2.numpy()])
        e_distance = result[0][0]

        if method == "euclidean":
            return e_distance
        else:
            return e_distance
    except Exception as e:
        print("Exception thrown during get distance", e)
        return 1000


def getTensor(embedding):
    if isinstance(embedding, torch.Tensor):
        tensor = embedding
    elif isinstance(embedding, np.ndarray) is True:
        tensor = torch.from_numpy(embedding)
    else:
        raise Exception("embedding = %s has to be either np.ndarray or torch.Tensor, not %s" % (embedding, type(embedding)))

    return tensor
