import argparse
import gzip
import hashlib
import json
import logging
from pathlib import Path
from tqdm import tqdm

logging_fmt="%(asctime)s %(levelname)s: %(message)s"
logging.basicConfig(format=logging_fmt)
logger=logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

def get_md5_hash(text:str)->str:
    return hashlib.md5(text.encode()).hexdigest()

def main(args):
    logger.info(args)

    wikipedia_data_filepath:str=args.wikipedia_data_filepath
    extract_root_dirname:str=args.extract_root_dirname

    extract_root_dir=Path(extract_root_dirname)
    extract_root_dir.mkdir(parents=True,exist_ok=True)

    logger.info("Wikipediaデータの行数を数えています...")

    num_wiki_data_lines=0
    with gzip.open(wikipedia_data_filepath,"r") as r:
        for line in r:
            num_wiki_data_lines+=1

    logger.info("行数: {}".format(num_wiki_data_lines))

    with gzip.open(wikipedia_data_filepath,"r") as r:
        for line in tqdm(r,total=num_wiki_data_lines):
            data=json.loads(line)

            title=data["title"]
            text=data["text"]

            title_hash=get_md5_hash(title)

            wiki_save_dir=extract_root_dir.joinpath(title_hash)
            wiki_save_dir.mkdir(exist_ok=True)

            title_file=wiki_save_dir.joinpath("title.txt")
            with title_file.open("w") as w:
                w.write(title)

            text_file=wiki_save_dir.joinpath("text.txt")
            with text_file.open("w") as w:
                w.write(text)

    logger.info("処理が完了しました")

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--wikipedia_data_filepath",type=str,default="../Data/all_entities.json.gz")
    parser.add_argument("--extract_root_dirname",type=str,default="../Data/Wikipedia")
    args=parser.parse_args()

    main(args)
