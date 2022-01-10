import argparse
import json
from tqdm import tqdm
from typing import Dict,List

def main(args):
    input_filepath:str=args.input_filepath
    output_filepath:str=args.output_filepath

    print("ファイルを読み込んでいます...")

    samples:Dict[str,List[str]]={}
    with open(input_filepath,"r") as r:
        for line in tqdm(r):
            data=json.loads(line)

            qid=data["qid"]
            article_title=data["article_title"]

            if qid in samples:
                article_titles:List[str]=samples[qid]
                article_titles.append(article_title)
            else:
                article_titles=[article_title]
                samples[qid]=article_titles

    print("結果をファイルに書き込んでいます...")

    with open(output_filepath,"w") as w:
        for qid,article_titles in samples.items():
            data={
                "qid":qid,
                "article_titles":article_titles
            }

            line=json.dumps(data,ensure_ascii=False)
            w.write(line)
            w.write("\n")

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("-i","--input_filepath",type=str,default="../Data/Reader/train_negative_samples.jsonl")
    parser.add_argument("-o","--output_filepath",type=str,default="../Data/Reader/negative_articles.jsonl")
    args=parser.parse_args()

    main(args)
