import argparse
import logging
import subprocess
from pathlib import Path
from tqdm import tqdm

logging_fmt="%(asctime)s %(levelname)s: %(message)s"
logging.basicConfig(format=logging_fmt)
logger=logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

def main(args):
    logger.info(args)

    wikipedia_data_root_dirname:str=args.wikipedia_data_root_dirname
    mecab_dictionary_dirname:str=args.mecab_dictionary_dirname
    buffer_size:int=args.buffer_size
    start_index:int=args.start_index
    end_index:int=args.end_index

    logger.info("Wikipedia記事を読み込む準備をしています...")

    wikipedia_data_root_dir=Path(wikipedia_data_root_dirname)
    wikipedia_data_dirs=wikipedia_data_root_dir.glob("*")
    wikipedia_data_dirs=list(wikipedia_data_dirs)
    wikipedia_data_dirs.sort()

    logger.info("Wikipedia記事数: {}".format(len(wikipedia_data_dirs)))

    if end_index is None:
        end_index=len(wikipedia_data_dirs)

    logger.info("{}から{}のWikipedia記事に対して処理を行います".format(start_index,end_index))

    wikipedia_data_dirs=wikipedia_data_dirs[start_index:end_index]

    mecab_args=["mecab","-Owakati","-b",str(buffer_size),"-d",mecab_dictionary_dirname]

    for wikipedia_data_dir in tqdm(wikipedia_data_dirs):
        text_file:Path=wikipedia_data_dir.joinpath("text.txt")

        this_mecab_args=mecab_args.copy()
        this_mecab_args.append(str(text_file))

        proc=subprocess.run(this_mecab_args,stdout=subprocess.PIPE)
        ma_result=proc.stdout.decode("utf8")
        ma_result=ma_result.replace("\n","")

        wakati_file:Path=wikipedia_data_dir.joinpath("wakati.txt")
        with wakati_file.open("w") as w:
            w.write(ma_result)
        
if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--wikipedia_data_root_dirname",type=str,default="../../Data/Wikipedia")
    parser.add_argument("--mecab_dictionary_dirname",type=str,default="/usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd")
    parser.add_argument("--buffer_size",type=int,default=1024*1024*10)
    parser.add_argument("--start_index",type=int,default=0)
    parser.add_argument("--end_index",type=int)
    args=parser.parse_args()

    main(args)
