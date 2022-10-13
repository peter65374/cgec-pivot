"""
Python 3.6
Required:
sudo pip3 install torch
sudo pip3 install sentence-transformers
sudo pip3 install transformers
sudo pip3 install sklearn
sudo pip3 install spacy
sudo python3 -m spacy download en_core_web_sm
sudo python3 -m spacy download zh_core_web_sm
sudo python3 -m spacy download zh_core_web_lg
sudo python3 -m spacy download zh_core_web_trf
"""

import logging

import spacy

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("NER")
nlp = None


def ner_chinese(sentence):
    result = []
    doc = ner_chinese_spacy(sentence)
    for token in doc:
        result.append({"text": str(token.text), "pos": str(token.pos_), "tag": str(token.tag_)})
    return result


def ner_chinese_spacy(text):
    global nlp
    if nlp is None:
        logger.info("START - 加载 NER 模型")
        nlp = spacy.load("zh_core_web_trf")
        logger.info("FINISH - 加载 NER 模型")

    doc = nlp(text)
    return doc
