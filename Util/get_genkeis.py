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

    mecab_args=["mecab","-b",str(buffer_size),"-d",mecab_dictionary_dirname]

    for idx,wikipedia_data_dir in enumerate(tqdm(wikipedia_data_dirs)):
        if idx<start_index:
            continue
        if idx>=end_index:
            break

        text_file:Path=wikipedia_data_dir.joinpath("text.txt")

        this_mecab_args=mecab_args.copy()
        this_mecab_args.append(str(text_file))

        proc=subprocess.run(this_mecab_args,stdout=subprocess.PIPE)
        ma_result=proc.stdout.decode("utf8")
        lines=ma_result.splitlines()
        lines.pop()

        genkeis=[]
        for line in lines:
            details=line.split("\t")
            if len(details)!=2:
                continue

            genkei=details[1].split(",")[6]
            genkeis.append(genkei)

        genkeis_file:Path=wikipedia_data_dir.joinpath("genkeis.txt")
        with genkeis_file.open("w") as w:
            for genkei in genkeis:
                w.write("{} ".format(genkei))

    logger.info("処理が終了しました")

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--wikipedia_data_root_dirname",type=str,default="../Data/Wikipedia")
    parser.add_argument("--mecab_dictionary_dirname",type=str,default="/usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd")
    parser.add_argument("--buffer_size",type=int,default=1024*1024*10)
    parser.add_argument("--start_index",type=int,default=0)
    parser.add_argument("--end_index",type=int)
    args=parser.parse_args()

    main(args)
