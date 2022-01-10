import argparse
import json
import random
from tqdm import tqdm
from typing import List,Tuple

def load_samples(samples_filepath:str)->Tuple[List[str],List[str],List[str],List[str],List[List[str]],List[List[str]]]:
    qids=[]
    questions=[]
    answers=[]
    positive_article_titles=[]
    answer_ranges=[]
    negative_article_titles=[]

    with open(samples_filepath,"r") as r:
        for line in r:
            data=json.loads(line)

            qid=data["qid"]
            question=data["question"]
            answer=data["answer"]
            positive_article_title=data["positive_article_title"]
            this_answer_ranges=data["answer_ranges"]
            this_negative_article_titles=data["negative_article_titles"]

            qids.append(qid)
            questions.append(question)
            answers.append(answer)
            positive_article_titles.append(positive_article_title)
            answer_ranges.append(this_answer_ranges)
            negative_article_titles.append(this_negative_article_titles)

    return qids,questions,answers,positive_article_titles,answer_ranges,negative_article_titles

def format_num_negative_articles(
    negative_article_titles:List[List[str]],
    wikipedia_article_titles:List[str],
    num_negative_articles_per_sample:int):
    for i in tqdm(range(len(negative_article_titles))):
        if len(negative_article_titles[i])>num_negative_articles_per_sample:
            negative_article_titles[i]=negative_article_titles[i][:num_negative_articles_per_sample]
        elif len(negative_article_titles[i])<num_negative_articles_per_sample:
            num_articles_to_be_added=num_negative_articles_per_sample-len(negative_article_titles[i])

            while True:
                random_titles=random.sample(wikipedia_article_titles,k=num_articles_to_be_added)

                duplicate_flag=False
                for random_title in random_titles:
                    if random_title in negative_article_titles[i]:
                        duplicate_flag=True
                        break

                if duplicate_flag==False:
                    break

            negative_article_titles[i]+=random_titles

def main(args):
    input_filepath:str=args.input_filepath
    wikipedia_titles_filepath:str=args.wikipedia_titles_filepath
    output_filepath:str=args.output_filepath
    num_negative_articles_per_sample:int=args.num_negative_articles_per_sample

    print("サンプルを読み込んでいます...")
    qids,questions,answers,positive_article_titles,answer_ranges,negative_article_titles=load_samples(input_filepath)

    print("Wikipedia記事のタイトル一覧を読み込んでいます...")
    with open(wikipedia_titles_filepath,"r") as r:
        wikipedia_article_titles=r.read().splitlines()

    print("負例のフォーマットを整えています...")
    format_num_negative_articles(negative_article_titles,wikipedia_article_titles,num_negative_articles_per_sample)

    print("データをファイルに書き込んでいます...")
    with open(output_filepath,"w") as w:
        for qid,question,answer,positive_article_title,this_answer_ranges,this_negative_article_titles in tqdm(zip(
            qids,questions,answers,positive_article_titles,answer_ranges,negative_article_titles),total=len(qids)):
            data={
                "qid":qid,
                "question":question,
                "answer":answer,
                "positive_article_title":positive_article_title,
                "answer_ranges":this_answer_ranges,
                "negative_article_titles":this_negative_article_titles
            }

            line=json.dumps(data,ensure_ascii=False)
            w.write(line)
            w.write("\n")

    print("処理が完了しました")

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("-i","--input_filepath",type=str,default="../Data/Reader/train_samples.jsonl")
    parser.add_argument("--wikipedia_titles_filepath",type=str,default="../Data/wikipedia_titles.txt")
    parser.add_argument("-o","--output_filepath",type=str,default="../Data/Reader/train_samples_2.jsonl")
    parser.add_argument("-n","--num_negative_articles_per_sample",type=int,default=20)
    args=parser.parse_args()

    main(args)
