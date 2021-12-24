import argparse
import json
import logging
from typing import List,Tuple

logging_fmt="%(asctime)s %(levelname)s: %(message)s"
logging.basicConfig(format=logging_fmt)
logger=logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

def load_samples(
    samples_filepath:str,
    limit_num_answer_ranges:int)->Tuple[List[str],List[str],List[str],List[str],List[List[str]]]:
    qids=[]
    questions=[]
    answers=[]
    article_titles=[]
    answer_ranges=[]

    with open(samples_filepath,"r") as r:
        for line in r:
            data=json.loads(line)

            qid=data["qid"]
            question=data["question"]
            answer=data["answer"]
            article_title=data["article_title"]
            this_answer_ranges=data["answer_ranges"][:limit_num_answer_ranges]

            qids.append(qid)
            questions.append(question)
            answers.append(answer)
            article_titles.append(article_title)
            answer_ranges.append(this_answer_ranges)

    return qids,questions,answers,article_titles,answer_ranges

def main(args):
    logger.info(args)

    input_filepath:str=args.input_filepath
    output_filepath:str=args.output_filepath
    limit_num_answer_ranges:int=args.limit_num_answer_ranges

    qids=[]
    questions=[]
    answers=[]
    article_titles=[]
    answer_ranges=[]

    logger.info("ファイルを読み込んでいます...")

    with open(input_filepath,"r") as r:
        for line in r:
            data=json.loads(line)

            qid=data["qid"]
            question=data["question"]
            answer=data["answer"]
            article_title=data["article_title"]
            this_answer_ranges=data["answer_ranges"][:limit_num_answer_ranges]

            qids.append(qid)
            questions.append(question)
            answers.append(answer)
            article_titles.append(article_title)
            answer_ranges.append(this_answer_ranges)

    logger.info("ファイルを作成しています...")

    with open(output_filepath,"w") as w:
        for qid,question,answer,article_title,this_answer_ranges in zip(qids,questions,answers,article_titles,answer_ranges):
            data={
                "qid":qid,
                "question":question,
                "answer":answer,
                "article_title":article_title,
                "answer_ranges":this_answer_ranges
            }

            line=json.dumps(data,ensure_ascii=False)
            w.write(line)
            w.write("\n")

    logger.info("処理が完了しました")

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("-i","--input_filepath",type=str)
    parser.add_argument("-o","--output_filepath",type=str)
    parser.add_argument("-n","--limit_num_answer_ranges",type=int,default=1)
    args=parser.parse_args()

    main(args)
