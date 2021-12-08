import argparse
import hashlib
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

def get_md5_hash(text:str)->str:
    return hashlib.md5(text.encode()).hexdigest()

def main(args):
    logger.info(args)

    src_data_filepath:str=args.src_data_filepath
    wikipedia_root_dirname:str=args.wikipedia_root_dirname
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

    logger.info("Wikipedia記事データを読み込む準備をしています...")

    wikipedia_root_dir=Path(wikipedia_root_dirname)
    wikipedia_dirs=wikipedia_root_dir.glob("*")
    wikipedia_dirs=list(wikipedia_dirs)

    logger.info("Wikipedia記事数: {}".format(len(wikipedia_dirs)))

    logger.info("サンプルを作成しています...")

    samples:List[Dict[str,str,str,bool]]=[]

    for src_data in tqdm(src_data_list):
        question=src_data["question"]
        answers=src_data["answers"]

        for answer in answers:
            title_hash=get_md5_hash(answer)
            wikipedia_dir=wikipedia_root_dir.joinpath(title_hash)
            if not wikipedia_dir.exists():
                #logger.warn("存在しないWikipedia記事名です: {}".format(answer))
                continue

            wikipedia_text_file=wikipedia_dir.joinpath("text.txt")
            with wikipedia_text_file.open("r") as r:
                wikipedia_text=r.read()

                sample={
                    "question":question,
                    "answer":answer,
                    "wikipedia_text":wikipedia_text,
                    "is_corresponding_text":True
                }
                samples.append(sample)

            if random.random()<create_negative_prob:
                rnd_wikipedia_dir=random.choice(wikipedia_dirs)
                rnd_wikipedia_text_file=rnd_wikipedia_dir.joinpath("text.txt")

                with rnd_wikipedia_text_file.open("r") as r:
                    rnd_wikipedia_text=r.read()

                    sample={
                        "question":question,
                        "answer":answer,
                        "wikipedia_text":rnd_wikipedia_text,
                        "is_corresponding_text":False
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
    parser.add_argument("--wikipedia_root_dirname",type=str,default="../Data/Wikipedia")
    parser.add_argument("--create_negative_prob",type=float,default=0.5)
    parser.add_argument("--samples_save_dirpath",type=str,default="../Data")
    parser.add_argument("--samples_save_filename",type=str,default="train_samples.jsonl")
    args=parser.parse_args()

    main(args)
