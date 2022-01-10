import argparse
import hashlib
import json
from pathlib import Path

def get_md5_hash(text:str)->str:
    return hashlib.md5(text.encode()).hexdigest()

def main(args):
    retrieve_result_filepath:str=args.retrieve_result_filepath
    wikipedia_data_root_dirname:str=args.wikipedia_data_root_dirname
    target_qid:str=args.target_qid
    target_text:str=args.target_text

    top_k_titles=None
    with open(retrieve_result_filepath,"r") as r:
        for line in r:
            data=json.loads(line)

            qid=data["qid"]
            if qid==target_qid:
                top_k_titles=data["top_k_titles"]
                break

    if top_k_titles is None:
        print("指定された問題は存在しません")
        return

    wikipedia_data_root_dir=Path(wikipedia_data_root_dirname)

    positive_articles=[]
    for top_k_title in top_k_titles:
        title_hash=get_md5_hash(top_k_title)
        text_file=wikipedia_data_root_dir.joinpath(title_hash,"text.txt")

        with text_file.open("r") as r:
            context=r.read()

        if target_text in context:
            positive_articles.append(top_k_title)

    if len(positive_articles)==0:
        print("指定されたテキストは出現しません")
    else:
        print(positive_articles)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--retrieve_result_filepath",type=str,default="../Data/Retriever/dev_top_ks.jsonl")
    parser.add_argument("--wikipedia_data_root_dirname",type=str,default="../Data/Wikipedia")
    parser.add_argument("-q","--target_qid",type=str)
    parser.add_argument("-t","--target_text",type=str)
    args=parser.parse_args()

    main(args)
