import argparse
import json
from tqdm import tqdm
from typing import Dict,List,Tuple

def load_positive_samples(positive_samples_filepath:str)->Tuple[List[str],List[str],List[str],List[str],List[List[str]]]:
    qids=[]
    questions=[]
    answers=[]
    article_titles=[]
    answer_ranges=[]

    with open(positive_samples_filepath,"r") as r:
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

def load_negative_articles(negative_articles_filepath:str)->Dict[str,List[str]]:
    samples:Dict[str,List[str]]={}

    with open(negative_articles_filepath,"r") as r:
        for line in r:
            data=json.loads(line)

            qid=data["qid"]
            article_titles=data["article_titles"]

            samples[qid]=article_titles

    return samples

def main(args):
    positive_samples_filepath:str=args.positive_samples_filepath
    negative_articles_filepath:str=args.negative_articles_filepath
    output_filepath:str=args.output_filepath

    print("正例を読み込んでいます...")
    qids,questions,answers,article_titles,answer_ranges=load_positive_samples(positive_samples_filepath)

    print("負例を読み込んでいます...")
    negative_articles=load_negative_articles(negative_articles_filepath)

    print("データを作成しています...")
    with open(output_filepath,"w") as w:
        for qid,question,answer,article_title,this_answer_ranges in tqdm(zip(qids,questions,answers,article_titles,answer_ranges),total=len(qids)):
            if qid in negative_articles:
                this_negative_articles=negative_articles[qid]

                data={
                    "qid":qid,
                    "question":question,
                    "answer":answer,
                    "positive_article_title":article_title,
                    "answer_ranges":this_answer_ranges,
                    "negative_article_titles":this_negative_articles
                }

                line=json.dumps(data,ensure_ascii=False)
                w.write(line)
                w.write("\n")

    print("処理が完了しました")

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--positive_samples_filepath",type=str,default="../Data/Reader/train_positive_samples.jsonl")
    parser.add_argument("--negative_articles_filepath",type=str,default="../Data/Reader/negative_articles.jsonl")
    parser.add_argument("--output_filepath",type=str,default="../Data/Reader/train_samples.jsonl")
    args=parser.parse_args()

    main(args)
