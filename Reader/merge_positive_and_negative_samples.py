import argparse
import json
from tqdm import tqdm
from typing import Dict,List,Tuple

def load_positive_samples(samples_filepath:str)->Tuple[List[str],List[str],List[str],List[str],List[List[str]]]:
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
            this_answer_ranges=data["answer_ranges"]

            qids.append(qid)
            questions.append(question)
            answers.append(answer)
            article_titles.append(article_title)
            answer_ranges.append(this_answer_ranges)

    return qids,questions,answers,article_titles,answer_ranges

def load_negative_samples(samples_filepath:str,max_num_negative_samples_per_question:int)->Dict[str,List[str]]:
    negative_samples:Dict[str,List[str]]={}

    with open(samples_filepath,"r") as r:
        for line in r:
            data=json.loads(line)

            qid=data["qid"]
            article_title=data["article_title"]

            if qid not in negative_samples:
                negative_samples[qid]=[article_title]
            else:
                article_titles=negative_samples[qid]
                if len(article_titles)>max_num_negative_samples_per_question:
                    continue

                article_titles.append(article_title)

    return negative_samples

def main(args):
    positive_samples_filepath:str=args.positive_samples_filepath
    negative_samples_filepath:str=args.negative_samples_filepath
    output_filepath:str=args.output_filepath
    max_num_negative_samples_per_question:int=args.max_num_negative_samples_per_question

    print("正例を読み込んでいます...")
    qids,questions,answers,article_titles,answer_ranges=load_positive_samples(positive_samples_filepath)

    print("負例を読み込んでいます...")
    negative_samples=load_negative_samples(negative_samples_filepath,max_num_negative_samples_per_question)

    print("正例と負例を合成しています...")
    o_qids=[]
    o_questions=[]
    o_answers=[]
    o_article_titles=[]
    o_answer_ranges=[]

    for qid,question,answer,article_title,this_answer_ranges in tqdm(zip(qids,questions,answers,article_titles,answer_ranges),total=len(qids)):
        #正例
        o_qids.append(qid)
        o_questions.append(question)
        o_answers.append(answer)
        o_article_titles.append(article_title)
        o_answer_ranges.append(this_answer_ranges)

        #負例
        if qid not in negative_samples:
            continue

        negative_article_titles=negative_samples[qid]
        for negative_article_title in negative_article_titles:
            o_qids.append(qid)
            o_questions.append(question)
            o_answers.append(answer)
            o_article_titles.append(negative_article_title)
            o_answer_ranges.append(["0-0"])

    print("データをファイルに書き込んでいます...")
    with open(output_filepath,"w") as w:
        for qid,question,answer,article_title,this_answer_ranges in zip(o_qids,o_questions,o_answers,o_article_titles,o_answer_ranges):
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

    print("処理が完了しました")

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("-p","--positive_samples_filepath",type=str,default="../Data/Reader/train_positive_samples.jsonl")
    parser.add_argument("-x","--negative_samples_filepath",type=str,default="../Data/Reader/train_negative_samples.jsonl")
    parser.add_argument("-o","--output_filepath",type=str)
    parser.add_argument("-n","--max_num_negative_samples_per_question",type=int)
    args=parser.parse_args()

    main(args)
