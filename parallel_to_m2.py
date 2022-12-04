import os
from pathlib import Path
import argparse
# from collections import Counter
# from collections import defaultdict
from multiprocessing import Pool
from tqdm import tqdm
# import torch
from opencc import OpenCC
from annotator import Annotator
from tokenizer import Tokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

annotator, sentence_to_tokenized = None, None
cc = OpenCC("t2s")
granularity = 'char'
device = 0  # num of GPU
batch_size = 128
segmented = False  # 原句没有预先segmented by whitespace
multi_cheapest_strategy = 'all'
no_simplified = True  # 全部转成简体中文


def annotate(line):
    """
    :param line:
    :return:
    """
    sent_list = line.split("\t")[1:]
    source = sent_list[0]
    if segmented:
        source = source.strip()
    else:
        source = "".join(source.strip().split())
    output_str = ""
    for idx, target in enumerate(sent_list[1:]):
        try:
            if segmented:
                target = target.strip()
            else:
                target = "".join(target.strip().split())
            if not no_simplified:
                target = cc.convert(target)
            source_tokenized, target_tokenized = sentence_to_tokenized[source], sentence_to_tokenized[target]
            out, cors = annotator(source_tokenized, target_tokenized, idx)
            if idx == 0:
                output_str += "".join(out[:-1])
            else:
                output_str += "".join(out[1:-1])
        except Exception:
            raise Exception
    return output_str

def main(inputfile, outputfile):
    # tokenizer = Tokenizer(args.granularity, args.device, args.segmented)
    tokenizer = Tokenizer(granularity, device, segmented)
    global annotator, sentence_to_tokenized
    annotator = Annotator.create_default(granularity, multi_cheapest_strategy)
    lines = open(inputfile, "r", encoding="utf-8").read().strip().split("\n")  # format: id src tgt1 tgt2...

    with open(outputfile, "w", encoding="utf-8") as f:
        count = 0
        sentence_set = set()
        sentence_to_tokenized = {}
        for line in lines:
            sent_list = line.split("\t")[1:]
            for idx, sent in enumerate(sent_list):
                if segmented:
                    sent = sent.strip()
                else:
                    sent = "".join(sent.split()).strip()  # ???
                if idx >= 1:
                    if not no_simplified:
                        sentence_set.add(cc.convert(sent))  # 如果input中有非简体中文，则用OpenCC转换
                    else:
                        sentence_set.add(sent)
                else:
                    sentence_set.add(sent)
        batch = []
        for sent in tqdm(sentence_set):
            count += 1
            if sent:
                batch.append(sent)
            if count % batch_size == 0:  # batch_size default is 128, 每128个句子批处理一次。
                results = tokenizer(batch)
                for s, r in zip(batch, results):
                    sentence_to_tokenized[s] = r  # Get tokenization map.
                batch = []
        if batch:  # last batch不足128数量的批处理一次。
            results = tokenizer(batch)
            for s, r in zip(batch, results):
                sentence_to_tokenized[s] = r  # Get tokenization map.
    
        # 单进程模式
        for line in tqdm(lines):
            ret = annotate(line)
            f.write(ret)
            f.write("\n") 

        # 多进程模式：仅在Linux环境下测试，建议在linux服务器上使用
        # with Pool(args.worker_num) as pool:
        #     for ret in pool.imap(annotate, tqdm(lines), chunksize=8):
        #         if ret:
        #             f.write(ret)
        #             f.write("\n")


if __name__ == "__main__":
    '''parser = argparse.ArgumentParser(description="Choose input file to annotate")
    parser.add_argument("-f", "--file", type=str, required=True, help="Input parallel file")
    parser.add_argument("-o", "--output", type=str, help="Output file", required=True)
    parser.add_argument("-b", "--batch_size", type=int, help="The size of batch", default=128)
    parser.add_argument("-d", "--device", type=int, help="The ID of GPU", default=0)
    parser.add_argument("-w", "--worker_num", type=int, help="The number of workers", default=16)
    parser.add_argument("-g", "--granularity", type=str, help="Choose char-level or word-level evaluation", default="char")
    parser.add_argument("-m", "--merge", help="Whether merge continuous replacement/deletion/insertion", action="store_true")
    parser.add_argument("-s", "--multi_cheapest_strategy", type=str, choices=["first", "all"], default="all")
    parser.add_argument("--segmented", help="Whether tokens have been segmented", action="store_true")  # char模式支持提前token化，用空格隔开
    parser.add_argument("--no_simplified", help="Whether simplifying chinese", action="store_true")  # 将所有corrections转换为简体中文
    args = parser.parse_args()'''
    abspath1 = Path(__file__).parent.joinpath('data/query_input.txt')
    abspath2 = Path(__file__).parent.joinpath('data/query_input_m2.txt')
    main(abspath1, abspath2)
