"""
Python 3.6
Required:
sudo pip3 install torch
sudo pip3 install sentence-transformers
sudo pip3 install transformers
sudo pip3 install sklearn

torch is Pytorch framework from pytorch.org
transformers is huggingface transformers lib from huggingface.com
sklearn is sci-kit learn data processing package from scikit-learn.org
sentence-transformers is multilingual sentence embedding from sbert.net made by UKP Lab

"""

import logging
from typing import Text, Any

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import euclidean_distances

logger = logging.getLogger("SIMILARITY")
model = None


#获取相似性 和 欧几里得距离两个数值。大致可以理解为这两个对句子语义近似判读是成反比的关系。
def getSimilarityAndDistance(sentence1, sentence2):
    global model
    if model is None:
        logger.info("START - 加载 SIMILARITY 模型")
        model = SentenceTransformer('distiluse-base-multilingual-cased-v2') #初始化model
        logger.info("FINISH - 加载 SIMILARITY 模型")

    embedding1 = model.encode(sentence1)
    embedding2 = model.encode(sentence2)
    return getSimilarity(embedding1, embedding2), getDistance(embedding1, embedding2)


def getSimilarity(embedding1, embedding2, method="cos"):
    # type: (Any, Any, Text) -> float
    try:
        tensor_1 = getTensor(embedding1)
        tensor_2 = getTensor(embedding2)

        #get语义similarity
        #CosineSimilarity函数详见 https://pytorch.org/docs/stable/nn.html#distance-functions
        #torch.nn的distance func还有 nn.PairwiseDistance()，当参数=2的时候，pairwise的p-norm距离就是欧几里得距离
    
        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)  #？？这里dim为什么是0，不是default value=1??
        result = cos(tensor_1, tensor_2)
        similarity = result.item()

        if method == "cos":
            return similarity
        else:
            return similarity
    except Exception as e:
        print("Exception thrown during get similarity", e)
        return 0


def getDistance(embedding1, embedding2, method="euclidean"):
    # type: (Any, Any, Text) -> float
    try:
        tensor_1 = getTensor(embedding1)
        tensor_2 = getTensor(embedding2)

        # get euclidean distance
        result = euclidean_distances([tensor_1.numpy()], [tensor_2.numpy()]) #这里使用了sklearn的pairwise,p=2时欧几里得距离函数
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
