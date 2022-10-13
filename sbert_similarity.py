"""
Python 3.6
Pytorch 1.6.0
transformer 4.6.0
Required:
sudo pip3 install torch
sudo pip3 install sentence-transformers
sudo pip3 install transformers

torch is Pytorch framework from pytorch.org
transformers is huggingface transformers lib from huggingface.com
sentence-transformers is multilingual sentence embedding from sbert.net made by UKP Lab

这里我们不使用sklearn去计算欧几里得距离，而用sbert.net的sentence-transformers自带的similarity函数来进行计算，
或者使用torch.nn.pairwise(p=2)来计算欧式距离

"""

import logging
from typing import Text, Any

#import numpy as np
#import torch
from sentence_transformers import SentenceTransformer, util
#from sklearn.metrics.pairwise import euclidean_distances


logger = logging.getLogger("SBERT.net SIMILARITY")
model = None

#model = SentenceTransformer('all-MiniLM-L6-v2')
model = SentenceTransformer('uer/sbert-base-chinese-nli') # model地址在 https://huggingface.co/uer/sbert-base-chinese-nli


#Our sentences we like to encode
sentences = [
    '妻子为什么很得意？',
    '亡羊补牢的故事告诉我们什么？', 
    'The quick brown fox jumps over the lazy dog.',
    '妻子很郁闷.',
    '妻子很生气.',
    'this is a happy dog.',
    '猫很开心',
    '妻子在吃东西',
    '妻子不开心',
    '妻子很开心',
    '妻子很高兴',
    '妻子给酒里掺水了。'
    ]

#Sentences are encoded by calling model.encode()
embeddings = model.encode(sentences)


#Compute cosine similarity between all pairs
cos_sim = util.cos_sim(embeddings, embeddings)
#cos_sim = util.pairwise_dot_score(embeddings, embeddings)

#Add all pairs to a list with their cosine similarity score
all_sentence_combinations = []
for i in range(len(cos_sim)-1):
    for j in range(i+1, len(cos_sim)):
        all_sentence_combinations.append([cos_sim[i][j], i, j])

#Sort list by the highest cosine similarity score
all_sentence_combinations = sorted(all_sentence_combinations, key=lambda x: x[0], reverse=True)

print("Top-5 most similar pairs:")
for score, i, j in all_sentence_combinations[0:5]:
    print("{} \t {} \t {:.4f}".format(sentences[i], sentences[j], cos_sim[i][j]))



'''
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

'''