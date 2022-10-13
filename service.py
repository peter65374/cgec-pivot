import logging

import nlp_ner
import nlp_similarity
import repo

logger = logging.getLogger("SERVICE")


def featureDescribe(features):
    _result = ''
    for i in range(0, len(features)):
        _result += features[i]['type'] + ': ' + features[i]['content'] + "\n"
    return _result


def nerDescribe(nerList):
    _result = ''
    for i in range(0, len(nerList)):
        _result += nerList[i]['text'] + ': ' + str(nerList[i]['pos']) + ',' + str(nerList[i]['tag']) + "\n"
    return _result


def containsStr(sentence, words):
    for word in words:
        if sentence.find(word) != -1:
            return True
    return False


def doFeatureScore(sentence, features):
    total = len(features)
    success = 0
    for i in range(0, len(features)):
        words = features[i]['content'].split('/')
        if containsStr(sentence, words):
            success = success + 1

    score = 100.0 / total * success
    return "\n关键词分数：" + str(score) + " 共" + str(total) + "个要素" + " 达成" + str(success) + "个要素"


def computeScore(sentence, features):
    """
    一、六要素（时间、地点、人物、事件等）【加分项】
        使用词性抽取算法抽取出不同词性的分类
        对比题库题目的6要素词汇要求，一般每个题目有 4-10 个关键词要求。合计100分的话，六要素每个类别的第一个错误扣 20 分，第二个及以后扣 10 分。一共 6 个大类，如果某个类别没有要求，不扣分。合计 100 分，扣完为止。

    二、错字扣分【减分项】

    三、基础语法（的/地/得)扣分【减分项】

    四、句式要求【加分项】
        如果有句式要求，做句式关键词对比，例如，因为... 所以 .... ；只要.... 句式符合要求不扣分，否则扣句式分。

    五、时间顺序、空间顺序【加分项】
        时间顺序和空间顺序检测方法，还未知，需要研发
    """
    scoreDesc = doFeatureScore(sentence, features)
    return scoreDesc


def buildResult(sentence, answers, features):
    result = ''
    scoreDescribe = computeScore(sentence, features)
    result = result + '\n分数：' + scoreDescribe + '\n'
    result = result + '\n要素：\n' + featureDescribe(features)
    result = result + "\n句子命名实体识别（NER）：\n" + sentence + '\n' + nerDescribe(nlp_ner.ner_chinese(sentence))

    for answer in answers:
        similarity = nlp_similarity.getSimilarityAndDistance(sentence, answer)
        result = result + '\n============================================================================='
        result = result + '\n1、标准答案：' + answer + '\n'
        result = result + "\n2、句子相似度：" + "相似度: " + str(similarity[0]) + "\t距离: " + str(similarity[1]) + '\n'
        result = result + "\n3、答案命名实体识别（NER）：\n" + nerDescribe(nlp_ner.ner_chinese(answer)) + '\n'
    return result


def doCompute(_id, sentence):
    item = repo.getById(_id)
    if item is None:
        return 'ID错误'

    answers = item['answers']
    if answers is None or len(answers) == 0:
        return '没有配置答案'

    features = item['features']
    if features is None or len(features) == 0:
        return '没有配置关键词'

    return buildResult(sentence, answers, features)
