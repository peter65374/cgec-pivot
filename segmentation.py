
from config import config_manager
from aigent2.engine.webhook.webhook import webhook
from typing import Text

import stanza
import json
import os


def main():

    texts = [
        "我不知道，你想要什么？今天天气不错。",
        "今天天气不错，我们出去散步吧。我再看一下价格，真不知道要买什么，你觉得呢？帮我叫个taxi怎么样？"
    ]

    print("loading lib ...")
    pipeline = stanza.Pipeline(lang="zh")

    for text in texts:
        print("processing text = '%s" % text)
        doc = pipeline(doc=text)
        sentences = []
        for sentence in doc.sentences:
            sentences.append(sentence.text)

        print("sentences (%s) => %s \n\n\n" % (len(sentences), sentences))

    print("done")


if __name__ == '__main__':
    main()
