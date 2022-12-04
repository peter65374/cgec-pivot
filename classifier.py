from pathlib import Path
from collections import namedtuple
# import os
from char_smi import CharFuncs
from pypinyin import pinyin, Style

Correction = namedtuple(
    "Correction",
    [
        "op",
        "toks",
        "inds",
    ],
) 
# file_path = os.path.dirname(os.path.abspath(__file__))
# char_smi = CharFuncs(os.path.join(file_path.replace("modules", ""), 'data/char_meta.txt'))
file_path = Path(__file__).parent.joinpath('data/char_meta.txt')
char_smi = CharFuncs(file_path)


# 这里有个悖论，seq2edit纠错model去分类拼写错误，能反向输入到自身model train里面去强化model么？ 
# 或者说单独训练一个错误分类器是否有意义呢？
def check_spell_error(src_span: str,
                      tgt_span: str,
                      threshold: float = 0.8) -> bool:
    if len(src_span) != len(tgt_span):
        return False
    src_chars = [ch for ch in src_span]
    tgt_chars = [ch for ch in tgt_span]
    if sorted(src_chars) == sorted(tgt_chars):  # 词内部字符异位
        return True
    for src_char, tgt_char in zip(src_chars, tgt_chars):
        if src_char != tgt_char:
            if src_char not in char_smi.data or tgt_char not in char_smi.data:
                return False
            v_sim = char_smi.shape_similarity(src_char, tgt_char)
            p_sim = char_smi.pronunciation_similarity(src_char, tgt_char)
            # 简单加法？似乎应该用 sqrt((v_sim*v_sim + p_sim*p_sim)/2) normalized to 0~1数值判断threshold
            if v_sim + p_sim < threshold and not (
                    set(pinyin(src_char, style=Style.NORMAL, heteronym=True)[0]) & set(pinyin(tgt_char, style=Style.NORMAL, heteronym=True)[0])):
                return False
    return True

class Classifier:
    """
    错误类型分类器
    """
    def __init__(self,
                 granularity: str = "word"):

        self.granularity = granularity
    
    '''
        POS Tags 中科院计算所ictclas兼容Tag: Jieba/Baidu/Paddle
        标签	含义	标签	含义	标签	含义	标签	含义
        n	普通名词	f	方位名词	s	处所名词	t	时间
        nr	人名	ns	地名	nt	机构名	nw	作品名
        nz	其他专名	v	普通动词	vd	动副词	vn	名动词
        a	形容词	ad	副形词	an	名形词	d	副词
        m	数量词	q	量词	r	代词	p	介词
        c	连词	u	助词	xc	其他虚词	w	标点符号
        PER	人名	LOC	地名	ORG	机构名	TIME	时间
        国标863 Pos Tag:
        Tag	Description	Example	Tag	Description	Example
        a	adjective	美丽	ni	organization name	保险公司
        b	other noun-modifier	大型, 西式	nl	location noun	城郊
        c	conjunction	和, 虽然	ns	geographical name	北京
        d	adverb	很	nt	temporal noun	近日, 明代
        e	exclamation	哎	nz	other proper noun	诺贝尔奖
        g	morpheme	茨, 甥	o	onomatopoeia	哗啦
        h	prefix	阿, 伪	p	preposition	在, 把
        i	idiom	百花齐放	q	quantity	个
        j	abbreviation	公检法	r	pronoun	我们
        k	suffix	界, 率	u	auxiliary	的, 地
        m	number	一, 第一	v	verb	跑, 学习
        n	general noun	苹果	wp	punctuation	，。！
        nd	direction noun	右侧	ws	foreign words	CPU
        nh	person name	杜甫, 汤姆	x	non-lexeme	萄, 翱
    '''
    @staticmethod
    def get_pos_type(pos):
        # if pos in {"n", "nd"}:
        if pos in {"n", "f"}:
            return "NOUN"
        # if pos in {"nh", "ni", "nl", "ns", "nt", "nz"}:
        if pos in {"nr", "nt", "s", "ns", "t", "nz", "PER", "LOC", "ORG", "TIME"}:
            return "NOUN-NE"
        # if pos in {"v"}:
        if pos in {"v", "vd", "vn"}:
            return "VERB"
        # if pos in {"a", "b"}:
        if pos in {"a", "ad", "an"}:
            return "ADJ"
        if pos in {"c"}:
            return "CONJ"
        if pos in {"r"}:
            return "PRON"
        if pos in {"d"}:
            return "ADV"
        if pos in {"u"}:
            return "AUX"
        # if pos in {"k"}:  # Todo 后缀词比例太少，暂且分入其它
        #     return "SUFFIX"
        if pos in {"m"}:
            return "NUM"
        if pos in {"p"}:
            return "PREP"
        if pos in {"q"}:
            return "QUAN"
        # if pos in {"wp"}:
        if pos in {"w"}:
            return "PUNCT"
        return "OTHER"

    def __call__(self,
                 src,
                 tgt,
                 edits,
                 verbose: bool = False):
        """
        为编辑操作划分错误类型
        :param src: 输入句子信息（错误/正确均有可能）
        :param tgt: 正确句子信息
        :param edits: 编辑操作
        :param verbose: 是否打印信息
        :return: 划分完错误类型后的编辑操作
        """
        '''
        编辑动作 vs. 纠错类型 对照表
        编辑操作：
        T: Tranposition, 移位编辑
        I: Insertion, 插入编辑
        D: Deletion, 删除编辑
        S: Substituion 替换编辑
        纠错错误类型 
        W: Word Order 词序错误
        R: Redundant 冗余
        M: Missing 缺少
        S: 拼写或词汇错误
        '''
        results = []
        src_tokens = [x[0] for x in src]
        tgt_tokens = [x[0] for x in tgt]
        for edit in edits:
            error_type = edit[0]
            src_span = " ".join(src_tokens[edit[1]: edit[2]])
            tgt_span = " ".join(tgt_tokens[edit[3]: edit[4]])
            # print(tgt_span)
            cor = None
            if error_type[0] == "T":
                cor = Correction("W", tgt_span, (edit[1], edit[2]))
            elif error_type[0] == "D":
                if self.granularity == "word":  # 词级别可以细分错误类型
                    if edit[2] - edit[1] > 1:  # 词组冗余暂时分为OTHER
                        cor = Correction("R:OTHER", "-NONE-", (edit[1], edit[2]))
                    else:
                        pos = self.get_pos_type(src[edit[1]][1])
                        pos = "NOUN" if pos == "NOUN-NE" else pos
                        pos = "MC" if tgt_span == "[缺失成分]" else pos
                        cor = Correction("R:{:s}".format(pos), "-NONE-", (edit[1], edit[2]))
                else:  # 字级别可以只需要根据操作划分类型即可
                    cor = Correction("R", "-NONE-", (edit[1], edit[2]))
            elif error_type[0] == "I":
                if self.granularity == "word":  # 词级别可以细分错误类型
                    if edit[4] - edit[3] > 1:  # 词组丢失暂时分为OTHER
                        cor = Correction("M:OTHER", tgt_span, (edit[1], edit[2]))
                    else:
                        pos = self.get_pos_type(tgt[edit[3]][1])
                        pos = "NOUN" if pos == "NOUN-NE" else pos
                        pos = "MC" if tgt_span == "[缺失成分]" else pos
                        cor = Correction("M:{:s}".format(pos), tgt_span, (edit[1], edit[2]))
                else:  # 字级别可以只需要根据操作划分类型即可
                    cor = Correction("M", tgt_span, (edit[1], edit[2]))
            elif error_type[0] == "S":
                if self.granularity == "word":  # 词级别可以细分错误类型
                    if check_spell_error(src_span.replace(" ", ""), tgt_span.replace(" ", "")):
                        cor = Correction("S:SPELL", tgt_span, (edit[1], edit[2]))
                        # Todo 暂且不单独区分命名实体拼写错误
                        # if edit[4] - edit[3] > 1:
                        #     cor = Correction("S:SPELL:COMMON", tgt_span, (edit[1], edit[2]))
                        # else:
                        #     pos = self.get_pos_type(tgt[edit[3]][1])
                        #     if pos == "NOUN-NE":  # 命名实体拼写有误
                        #         cor = Correction("S:SPELL:NE", tgt_span, (edit[1], edit[2]))
                        #     else:  # 普通词语拼写有误
                        #         cor = Correction("S:SPELL:COMMON", tgt_span, (edit[1], edit[2]))
                    else:
                        if edit[4] - edit[3] > 1:  # 词组被替换暂时分为OTHER
                            cor = Correction("S:OTHER", tgt_span, (edit[1], edit[2]))
                        else:
                            pos = self.get_pos_type(tgt[edit[3]][1])
                            pos = "NOUN" if pos == "NOUN-NE" else pos
                            pos = "MC" if tgt_span == "[缺失成分]" else pos
                            cor = Correction("S:{:s}".format(pos), tgt_span, (edit[1], edit[2]))
                else:  # 字级别可以只需要根据操作划分类型即可
                    cor = Correction("S", tgt_span, (edit[1], edit[2]))
            results.append(cor)
        if verbose:
            print("========== Corrections ==========")
            for cor in results:
                print("Type: {:s}, Position: {:d} -> {:d}, Target: {:s}".format(cor.op, cor.inds[0], cor.inds[1], cor.toks))
        return results
