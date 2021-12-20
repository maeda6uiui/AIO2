import argparse
import hashlib
import json
import logging
from pathlib import Path
from tqdm import tqdm
from typing import List,Tuple

logging_fmt="%(asctime)s %(levelname)s: %(message)s"
logging.basicConfig(format=logging_fmt)
logger=logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

def get_md5_hash(text:str)->str:
    return hashlib.md5(text.encode()).hexdigest()

def load_retrieval_results(retrieval_results_filepath:str)->Tuple[List[List[str]],List[List[str]]]:
    answers=[]
    top_k_titles=[]

    with open(retrieval_results_filepath,"r") as r:
        for line in r:
            data=json.loads(line)

            this_answers=data["answers"]
            this_top_k_titles=data["top_k_titles"]

            answers.append(this_answers)
            top_k_titles.append(this_top_k_titles)

    return answers,top_k_titles

def is_answer_contained_in_retrieved_articles(
    answers:List[str],
    article_titles:List[str],
    wikipedia_data_root_dir:Path)->bool:
    for article_title in article_titles:
        title_hash=get_md5_hash(article_title)

        text_file=wikipedia_data_root_dir.joinpath(title_hash,"text.txt")
        with text_file.open("r") as r:
            text=r.read()

        for answer in answers:
            if answer in text:
                return True

    return False

def main(args):
    logger.info(args)

    retrieval_results_filepath:str=args.retrieval_results_filepath
    wikipedia_data_root_dirname:str=args.wikipedia_data_root_dirname
    results_save_filepath:str=args.results_save_filepath

    logger.info("処理を行う準備をしています...")

    wikipedia_data_root_dir=Path(wikipedia_data_root_dirname)

    answers,top_k_titles=load_retrieval_results(retrieval_results_filepath)

    logger.info("正解率の計算を行っています...")

    ks=[1,5,10,50,100]
    accuracies=[]

    for k in ks:
        logger.info("k={}".format(k))

        question_count=0
        correct_count=0
        for this_answers,this_top_k_titles in tqdm(zip(answers,top_k_titles),total=len(answers)):
            this_top_k_titles=this_top_k_titles[:k]

            contained=is_answer_contained_in_retrieved_articles(this_answers,this_top_k_titles,wikipedia_data_root_dir)
            if contained:
                correct_count+=1

            question_count+=1

        accuracy=correct_count/question_count*100
        accuracies.append(accuracy)

        logger.info(">正解率: {} %".format(accuracy))

    with open(results_save_filepath,"w") as w:
        for k,accuracy in zip(ks,accuracies):
            w.write("Acc@{} = {}\n".format(k,accuracy))

    logger.info("処理が完了しました")

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--retrieval_results_filepath",type=str)
    parser.add_argument("--wikipedia_data_root_dirname",type=str,default="../Data/Wikipedia")
    parser.add_argument("--results_save_filepath",type=str)
    args=parser.parse_args()

    main(args)
