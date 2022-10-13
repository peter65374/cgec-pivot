# aut0-cr2t
xc-pen homework auto correction policy.
整体策略的描述大致如下：


aut0-cr2t 模块描述


Sp1: 题目与答案数据获取
● 题干获取 (json, local/cloud)
● 题型获取与判定 (json, local/cloud)
● 标准答案获取 (json, local/cloud)
● 评价策略获取 (json, local/cloud)
● 学生答案提取（grid-pen ocr）

Sp2: 依据评价策略批改
语义理解批改（MRC任务）
● 语义批改 (similarity & distance dep.  torch/scikit-learn/etc.)
● BLEU-4 评价分 (to explore the official bleu-4 lib)
● ROUGE-L 分数加关键词策略权重，关键词覆盖率，关键词命中率 (to explore the official rouge lib)

出题人命题关键词批改
● 分词NER (ner dep. spacy)
● 对于批改关键词，引入同义词和反义词词典、成语词典 (to make a private dict.)

语法批改 （也可以基于bert或T5的mask策略转换成完形填空任务）
● 句式批改(to ask)
● 上下文错字提取(pycorrector)
● 上下文错词提取(pycorrector)
● 修饰词、动词推荐(to ask)
● 引入成语词典，诗词，好词好句，名人名句词典等外挂强化库(to make a private dict.)

句子改写 (可选）
● 类grammarly的句子改写的实现，同样含义的更好的句子表达
● 句子改写方向策略，分成公文党建、学生作文、非科学叙事写作、科学论文等不同的语句倾向性
● 短语改写（不构成一个完整sentence的短语的改写）

Sp3: 批改反馈策略
三段策略：
1. 语义准确度反馈(0-1 value)
2. 出题人关键词策略反馈(有权重的，命中结果true or false)
3. 语法准确度反馈(错字、错词、修饰词、动词、句式等语法纠错提示)

例子：
1. 答案回答的主题含义和方向大体正确（偏离题意），但是还不完整。
2. 其中缺少对于“亡羊补牢” “妻子悔恨” 等几个关键点的表达和阐述
3. （语法错误）此外，xxx，xxxx等几处，有错别字标红，xxxx的词汇运用可以改进，改善建议见蓝色词汇。


author: peter
ver: 0.1
date: 2022.10.13
