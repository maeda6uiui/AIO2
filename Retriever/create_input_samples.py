import argparse
import json
import logging
import random
from pathlib import Path
from tqdm import tqdm
from typing import Dict,List

logging_fmt="%(asctime)s %(levelname)s: %(message)s"
logging.basicConfig(format=logging_fmt)
logger=logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

def main(args):
    logger.info(args)

    src_data_filepath:str=args.src_data_filepath
    wikipedia_titles_filepath:str=args.wikipedia_titles_filepath
    create_negative_prob:float=args.create_negative_prob
    samples_save_dirpath:str=args.samples_save_dirpath
    samples_save_filename:str=args.samples_save_filename

    samples_save_dir=Path(samples_save_dirpath)
    samples_save_dir.mkdir(parents=True,exist_ok=True)

    logger.info("入力データを読み込んでいます...")

    src_data_list:List[Dict[str,List[str]]]=[]
    with open(src_data_filepath,"r") as r:
        for line in r:
            data=json.loads(line)

            question=data["question"]
            answers=data["answers"]

            src_data={
                "question":question,
                "answers":answers
            }
            src_data_list.append(src_data)

    logger.info("入力データ数: {}".format(len(src_data_list)))

    logger.info("Wikipedia記事のタイトルを読み込んでいます...")

    wikipedia_titles_file=Path(wikipedia_titles_filepath)
    with wikipedia_titles_file.open("r") as r:
        titles=r.read().splitlines()
        titles.pop()    #最終行は空文字列

    logger.info("Wikipedia記事数: {}".format(len(titles)))

    logger.info("サンプルを作成しています...")

    samples:List[Dict[str,str,bool]]=[]

    for src_data in tqdm(src_data_list):
        question=src_data["question"]
        answers=src_data["answers"]

        for answer in answers:
            sample={
                "question":question,
                "given_article":answer,
                "is_corresponding_article":True
            }
            samples.append(sample)

            if random.random()<create_negative_prob:
                rnd_wikipedia_title=random.choice(titles)

                sample={
                    "question":question,
                    "given_article":rnd_wikipedia_title,
                    "is_corresponding_article":False
                }
                samples.append(sample)

    logger.info("作成されたサンプルをファイルに書き込んでいます...")

    samples_save_file=samples_save_dir.joinpath(samples_save_filename)
    with samples_save_file.open("w") as w:
        for sample in samples:
            line=json.dumps(sample,ensure_ascii=False)
            w.write(line)
            w.write("\n")

    logger.info("処理が完了しました")

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--src_data_filepath",type=str,default="../Data/aio_02_train.jsonl")
    parser.add_argument("--wikipedia_titles_filepath",type=str,default="../Data/wikipedia_titles.txt")
    parser.add_argument("--create_negative_prob",type=float,default=0.5)
    parser.add_argument("--samples_save_dirpath",type=str,default="../Data")
    parser.add_argument("--samples_save_filename",type=str,default="train_samples.jsonl")
    args=parser.parse_args()

    main(args)
