import jieba
import jieba.posseg as jseg
from typing import List
from pypinyin import pinyin, Style, lazy_pinyin

class Tokenizer:
    """
    分词器
    """

    def __init__(self,
                 granularity: str = "word",
                 device: str = "cpu",
                 segmented: bool = False,
                 ) -> None:
        """
        构造函数
        :param mode: 分词模式，可选级别：字级别（char）、词级别（word）
        """
        jieba.cut('东方红太阳升') 
        if granularity == "word":
            jieba.add_word('[缺失成分]')
        self.segmented = segmented
        self.granularity = granularity
        if self.granularity == "word":
            self.tokenizer = self.split_word
        elif self.granularity == "char":
            self.tokenizer = self.split_char
        else:
            raise NotImplementedError

    def __repr__(self) -> str:
        return "{:s}\nMode:{:s}\n}".format(str(self.__class__.__name__), self.granularity)

    def __call__(self,
                 input_strings: List[str]
                 ) -> List:
        """
        分词函数
        :param input_strings: 需要分词的字符串列表
        :return: 分词后的结果列表，由元组组成，元组为(token,pos_tag,pinyin)的形式
        """
        if not self.segmented:
            input_strings = ["".join(s.split(" ")) for s in input_strings]
        results = self.tokenizer(input_strings)
        return results

    def split_char(self, input_strings: List[str]) -> List:
        """
        分字函数
        :param input_strings: 需要分字的字符串
        :return: 分字结果
        """
        results = []
        for input_string in input_strings:
            if not self.segmented:  # 如果没有被分字，就按照每个字符隔开（不考虑英文标点的特殊处理，也不考虑BPE），否则遵循原分字结果
                segment_string = " ".join([char for char in input_string])
            else:
                segment_string = input_string
                # print(segment_string)
            segment_string = segment_string.replace("[ 缺 失 成 分 ]", "[缺失成分]").split(" ")  # '[缺失成分]'当成一个单独的token
            results.append([(char, "unk", pinyin(char, style=Style.TONE, heteronym=False)[0]) for char in segment_string])
        return results

    def split_word(self, input_strings: List[str]) -> List:
        """
        分词函数
        :param input_strings: 需要分词的字符串
        :return: 分词结果
        """
        seg, pos = [], []
        if self.segmented:
            seg_input_strings = ["".join(s.split(" ")) for s in input_strings]
            # seg_input_strings = []
            # for input in input_strings:
            #    seg_input_strings.extend(input.split(" ")) 
            for input_string in seg_input_strings:
                res = jseg.lcut(input_string)
                word = [x.word for x in res]
                flag = [x.flag for x in res]
                # print("{} {}".format(word, flag))
                seg.append(word)
                pos.append(flag)
        else:
            for input_string in input_strings:
                res = jseg.lcut(input_string)
                word = [x.word for x in res]
                flag = [x.flag for x in res]
                # print("{} {}".format(word, flag))
                seg.append(word)
                pos.append(flag)
        # print(seg)
        # print(pos)
        result = []
        for s, p in zip(seg, pos):  # 返回三元组 (词, postag, 拼音)
            pinyin = [lazy_pinyin(word, style=Style.TONE) for word in s]
            result.append(list(zip(s, p, pinyin)))
        return result

if __name__ == "__main__":
    tokenizer = Tokenizer("word")
    input_strings = ["结巴是个优秀的分词工具", "我爱北京天安门"]
    result = tokenizer(input_strings)
    print(result)
