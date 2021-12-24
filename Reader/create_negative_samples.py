import argparse
import json
import random
from tqdm import tqdm
from typing import List,Tuple

def load_samples(samples_filepath:str)->Tuple[List[str],List[str],List[str]]:
    qids=[]
    questions=[]
    answers=[]
   
    with open(samples_filepath,"r") as r:
        for line in r:
            data=json.loads(line)

            qid=data["qid"]
            question=data["question"]
            answer=data["answers"][0]

            qids.append(qid)
            questions.append(question)
            answers.append(answer)

    return qids,questions,answers

def main(args):
    input_filepath:str=args.input_filepath
    wikipedia_titles_filepath:str=args.wikipedia_titles_filepath
    output_filepath:str=args.output_filepath
    num_negative_samples_per_question:int=args.num_negative_samples_per_question

    qids,questions,answers=load_samples(input_filepath)

    with open(wikipedia_titles_filepath,"r") as r:
        article_titles=r.read().splitlines()

    ng_qids:List[str]=[]
    ng_questions:List[str]=[]
    ng_answers:List[str]=[]
    ng_article_titles:List[str]=[]
    ng_answer_ranges:List[List[str]]=[]

    for qid,question,answer in tqdm(zip(qids,questions,answers),total=len(qids)):
        for i in range(num_negative_samples_per_question):
            ng_qids.append(qid)
            ng_questions.append(question)
            ng_answers.append(answer)

            random_article=random.choice(article_titles)
            ng_article_titles.append(random_article)

            ng_answer_ranges.append(["0-0"])

    with open(output_filepath,"w") as w:
        for ng_qid,ng_question,ng_answer,ng_article_title,this_ng_answer_ranges in tqdm(zip(
            ng_qids,ng_questions,ng_answers,ng_article_titles,ng_answer_ranges),total=len(ng_qids)):
            data={
                "qid":ng_qid,
                "question":ng_question,
                "answer":ng_answer,
                "article_title":ng_article_title,
                "answer_ranges":this_ng_answer_ranges
            }

            line=json.dumps(data,ensure_ascii=False)

            w.write(line)
            w.write("\n")

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("-i","--input_filepath",type=str,default="../Data/aio_02_train.jsonl")
    parser.add_argument("-w","--wikipedia_titles_filepath",type=str,default="../Data/wikipedia_titles.txt")
    parser.add_argument("-o","--output_filepath",type=str,default="../Data/Reader/negative_samples.jsonl")
    parser.add_argument("-n","--num_negative_samples_per_question",type=int,default=1)
    args=parser.parse_args()

    main(args)
