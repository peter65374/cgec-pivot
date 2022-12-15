# CGEC - Chinese Grammar Error Correction Toolkit
目录说明
Directory:
ChERRANT: from MuCGEC, 语法纠错结果的评估打分工具，Thanks Zhangyue. 
其中分词用的LTP的部分改成使用jieba分词，因为LTP是GPL licence，jieba是apache licence. 本项目希望使用MIT/Apache这种比较宽松License的基础上构建。

collocation: Word/Phrease Collocation, 中文短语搭配词典，使用不同方法来给与短语搭配。

sql_trial: SQLAlchemy的例子，建议直接查看Fastapi + SQLAlchemy + SQlite/MySQL/PostgreSQL的样例，数据存储量不高的时候可以直接用pandas + numpy，pickle compress格式，有需要的的话用sqlite单文件数据库，再需要部署并发环境的话，用P-SQL or MySQL，阿里云环境的话一般是MySQL. 如果MongoDB的话，换pymongo作为ORM的界面driver。

utils: 一些临时试验文件，可删除。