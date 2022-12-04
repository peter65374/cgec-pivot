'''preprocess the file to extract:
query (原句)
input (输入)
ref answer (参考答案)
'''
import json
from pathlib import Path
import pandas as pd


datafile = 'data/type5inputzl.xlsx'
datafile2 = 'data/query_input.txt'
abspath1 = Path(__file__).parent.joinpath(datafile)
abspath2 = Path(__file__).parent.joinpath(datafile2)

df1 = pd.read_excel(abspath1)
df2 = df1[["question_query", "input", "answer_json"]]
print('DataFrame Types:', df2.dtypes)
print(df2.info())
print(df2.shape)

query_input_ref = df2.to_dict('list')
query_org = query_input_ref['question_query']
input_list = query_input_ref['input']
ref_json_list = query_input_ref['answer_json']
# 清除query里面的特殊格式${input}
query_list = []
for _query in query_org:
    _query = _query.replace('${input}', '')
    _query.strip()
    query_list.append(_query)
# 清除ref answer json string里面的json format格式
ref_answer_list = []
for _answer in ref_json_list:
    l_answer = json.loads(_answer)
    ref_answer_list.append(l_answer[0][0])

with open(datafile2, 'w+', encoding='utf-8') as f:
    for x in range(len(query_list)):
        line = str(x+1) + '\t' + query_list[x] + '\t' + input_list[x] + '\n'
        f.write(line)

'''
# remove WPS/Numbers格式转换tsv带来的多余的引号。
tsvfile = 'data/type7answer3.tsv'
tsvfile2 = 'data/type7answer3_clean.tsv'

newline = []

with open(tsvfile2, "w", encoding="utf-8") as newf:
    with open(tsvfile, "r", encoding="utf-8") as f:
        for line in f:  # format: question_unique_id answer_json
            line = line.replace('"[', '[')
            line = line.replace(']"', ']')
            newline = line.replace('""', '"')
            newf.write(newline)
'''
