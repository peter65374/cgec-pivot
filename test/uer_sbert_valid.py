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
"""

# from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers import SentenceTransformer, util
# import numpy as np
# from transformers import AutoTokenizer
# from transformers import AutoModel

# tokenizer = AutoTokenizer.from_pretrained("uer/sbert-base-chinese-nli")
# model = AutoModel.from_pretrained("uer/sbert-base-chinese-nli")
model = SentenceTransformer('uer/sbert-base-chinese-nli') #model地址在 https://huggingface.co/uer/sbert-base-chinese-nli

# Pre-trained cross encoder
#model = CrossEncoder('uer/sbert-base-chinese-nli')
#model = CrossEncoder('cross-encoder/stsb-distilroberta-base')
#model = CrossEncoder('cross-encoder/nli-roberta-base')

sentences1 = [
    '妻子很高兴.',
    '妻子很高兴.',
    '妻子很高兴.'
    ]

sentences2 = [
    '诚信才是我们最值钱的东西',
    '妻子非常高兴',
    '妻子很开心',
    '妻子不高兴',
    '妻子为什么很得意？',
    '亡羊补牢的故事告诉我们什么？', 
    'The quick brown fox jumps over the lazy dog.',
    '妻子很郁闷.',
    '妻子很生气.',
    'this is a happy dog.',
    '猫很开心',
    '妻子在吃东西',
    '妻子很高兴',
    '妻子给酒里掺水了。'
    ]

#Compute embedding for both lists
embeddings1 = model.encode(sentences1, convert_to_tensor=True)
embeddings2 = model.encode(sentences2, convert_to_tensor=True)

#Compute cosine-similarities
cosine_scores = util.cos_sim(embeddings1[0], embeddings2)

#Output the pairs with their score
for i in range(len(sentences2)):
    print("Score: {:.4f} \t\t {} \t\t {} ".format(cosine_scores[0][i], sentences1[0], sentences2[i]))

'''

用crossencoder的方式调用的测试，不过似乎不需要，有点迷，不知道为什么要用crossencoder了，从performance来看uer finetune之后的model明显效果就很好
crossencoder的意义何在呢？

This example computes the score between a query and all possible
sentences in a corpus using a Cross-Encoder for semantic textual similarity (STS).
It output then the most similar sentences for the given query.

# We want to compute the similarity between the query sentence
query = '妻子很高兴.'

# With all sentences in the corpus
corpus = [
    '妻子为什么很得意？',
    '亡羊补牢的故事告诉我们什么？', 
    'The quick brown fox jumps over the lazy dog.',
    '妻子很郁闷.',
    '妻子很生气.',
    'this is a happy dog.',
    '妻子非常高兴',
    '猫很开心',
    '妻子在吃东西',
    '妻子不高兴',
    '妻子很开心',
    '妻子很高兴',
    '妻子给酒里掺水了。'
    ]

# So we create the respective sentence combinations
sentence_combinations = [[query, corpus_sentence] for corpus_sentence in corpus]

# Compute the similarity scores for these combinations
similarity_scores = model.predict(sentence_combinations)

# Sort the scores in decreasing order
sim_scores_argsort = reversed(np.argsort(similarity_scores))

# Print the scores
print("Query:", query)
for idx in sim_scores_argsort:
    print("{:.2f}\t{}".format(similarity_scores[idx], corpus[idx]))

'''