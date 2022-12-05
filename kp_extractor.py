"""
Python 3.7
Required:
"""
import argparse
import logging
from typing import List
import json
import clueai
import time
import requests  
import jieba.analyse

API_ACCESS_TOKEN = ''
TOKEN_URL = 'https://aip.baidubce.com/oauth/2.0/token'
API_KEY = ''  # 例子需要更换
SECRET_KEY = ''  # 例子需要更换

KEYWORD_URL = 'https://aip.baidubce.com/rpc/2.0/nlp/v1/txt_keywords_extraction'
KEYWORD_HEADER = 'application/json'

CLUE_API_KEY = ''
cl = None

kelogger = logging.getLogger("Keypoint Extractor")
kelogger.setLevel(logging.INFO)
logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s")


# print (json.dumps(response.json(), ensure_ascii=False, indent=4))
# body = json.dumps({"text": sentence}).encode("utf-8")


# Jieba keyword extract
def jiebaExtract(answer: List[str]):
    keypoints = []
    for ref_sentences in answer:
        for sentence in ref_sentences:
            # response = jieba.analyse.extract_tags(sentence, topK=5, withWeight=True, allowPOS=())
            response = jieba.analyse.textrank(sentence, topK=20, withWeight=True, allowPOS=('ns', 'n', 'vn', 'v')) 
            kelogger.info(response)
            if response:
                keypoints.append(response)
            else:
                keypoints.append([])
            # time.sleep(0.5)
    return keypoints


# Open file and extract keyword by Baidu BCE API
def keywordExtract(answer: List[str]):
    keypoints = []
    request_url = KEYWORD_URL + "?charset=UTF-8&access_token=" + API_ACCESS_TOKEN  # coding = utf-8, baidu default is GBK, 所以在URL参数里面要声明utf-8
    headers = {'Content-type': KEYWORD_HEADER}
    for sentence in answer:
        params = {"text": sentence, 'num': 4}
        # kelogger.info(params)
        response = requests.post(request_url, json=params, headers=headers)
        if response:
            kps = response.json()
            if "results" in kps:
                # kelogger.info(kps["results"])
                keypoints.append(kps["results"])
            else:
                keypoints.append(kps)
        time.sleep(1.02)
    return keypoints


# Open file and extract keyword by ClueAI API
def extractKey(answer: List[str]):
    keypoints = []
    global cl
    cl = clueai.Client(CLUE_API_KEY)   # initialize the Clueai Client with an API Key
    for sentence in answer:
        myprompt = '抽取关键词：\n{}\n关键词：'.format(sentence[0])
        # testprompt = '文本纠错：\n{}\n答案：\n'.format(sentence[0])
        kelogger.info('Prompt:%s', myprompt)
        kstr = cl.generate(
            model_name = 'clueai-large',  #ClueAI Prompt Large API
            prompt = myprompt,
            return_likelihoods = "GENERATION"
            )
        kelogger.info('keyword:%s', kstr.generations[0].text)
        keypoints.append(kstr.generations[0].text.split())
        time.sleep(0.3)
    return keypoints


def main(args):
    keypoint = []
    with open(args.file, "r", encoding="utf-8") as f:
        for line in f:  # format: question_unique_id answer_json
            try:
                idx,  answer_json = line.strip().split("\t")
                # kelogger.info(answer_json)
            except ValueError as e:
                print(e)
            try:
                answer = json.loads(answer_json)
                kp = keywordExtract(answer)  # use ClueAI or Baidu API to extract keypoint
                # kelogger.info(kp)
                res_line = '{}\t{}\n'.format(idx, json.dumps(kp, ensure_ascii=False))
                kelogger.info(res_line)
                keypoint.append(res_line)
            except ValueError:  # includes simplejson.decoder.JSONDecodeError
                print('JSONDecodeError: questionId: {} - answer is not JSON format, failed to decode it.'.format(idx))
        print(keypoint)
        with open(args.output, "w", encoding="utf-8") as outf:
            for line in keypoint:
                kelogger.info(line)
                outf.write(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Choose input file to extract")
    parser.add_argument("-f", "--file", type=str, required=True, help="Input answer file")
    parser.add_argument("-o", "--output", type=str, help="Output keyword file", required=True)
    args = parser.parse_args()
    main(args)
