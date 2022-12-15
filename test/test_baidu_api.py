import requests
import logging
import time
from typing import List
# coding = utf-8, baidu default is GBK, 所以在参数里面要声明utf-8

API_ACCESS_TOKEN = ''
TOKEN_URL = 'https://aip.baidubce.com/oauth/2.0/token'
API_KEY = ''  # 例子需要更换
SECRET_KEY = ''  # 例子需要更换

KEYWORD_URL = 'https://aip.baidubce.com/rpc/2.0/nlp/v1/txt_keywords_extraction'
KEYWORD_HEADER = 'application/json'

kelogger = logging.getLogger("Keypoint Extractor")
kelogger.setLevel(logging.INFO)
logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s")


# Open file and extract keyword by Baidu BCE API
def keywordExtract(answer: List[str]):
    keypoints = []
    request_url = KEYWORD_URL + "?charset=UTF-8&access_token=" + API_ACCESS_TOKEN
    headers = {'Content-type': KEYWORD_HEADER}
    for sentence in answer:
        params = {"text": sentence, 'num': 4}
        kelogger.info(params)
        response = requests.post(request_url, json=params, headers=headers)
        if response:
            kps = response.json()
            if "results" in kps:
                kelogger.info(kps["results"])
                keypoints.append(kps["results"])
            else:
                keypoints.append(kps)
        time.sleep(0.5)
    return keypoints

answer = [["解释说明"]]
kp = keywordExtract(answer)  # use ClueAI or Baidu API to extract keypoint
kelogger.info(kp)
