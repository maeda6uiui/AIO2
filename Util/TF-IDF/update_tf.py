import argparse
import logging
from pathlib import Path
from tqdm import tqdm

logging_fmt="%(asctime)s %(levelname)s: %(message)s"
logging.basicConfig(format=logging_fmt)
logger=logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

def main(args):
    logger.info(args)

    wikipedia_data_root_dirname:str=args.wikipedia_data_root_dirname
    idf_filepath:str=args.idf_filepath
    start_index:int=args.start_index
    end_index:int=args.end_index

    logger.info("TFファイルを更新する準備をしています...")

    wikipedia_data_root_dir=Path(wikipedia_data_root_dirname)
    wikipedia_data_dirs=wikipedia_data_root_dir.glob("*")
    wikipedia_data_dirs=list(wikipedia_data_dirs)
    wikipedia_data_dirs.sort()

    if end_index is None:
        end_index=len(wikipedia_data_dirs)

    logger.info("{}から{}のWikipedia記事に対して処理を行います".format(start_index,end_index))

    logger.info("IDFファイルを読み込んでいます...")

    genkeis=[]
    with open(idf_filepath,"r") as r:
        for line in r:
            if line=="":
                continue

            genkei,idf=line.split("\t")
            idf=float(idf)

            genkeis.append(genkei)

    logger.info("TFファイルの更新を行っています...")

    for idx,wikipedia_data_dir in enumerate(tqdm(wikipedia_data_dirs)):
        if idx<start_index:
            continue
        if idx>=end_index:
            break

        tf_file:Path=wikipedia_data_dir.joinpath("tf.txt")
        with tf_file.open("r") as r:
            lines=r.read().splitlines()

        tf_genkeis=[]
        for line in lines:
            tf_genkei=line.split("\t")[0]
            tf_genkeis.append(tf_genkei)

        #TFファイルに存在しない単語を追加する
        #存在しない単語の場合、TF=0とする
        tf_not_exist_genkeis=[]
        for genkei in genkeis:
            if genkei not in tf_genkeis:
                tf_not_exist_genkeis.append(genkei)

        with tf_file.open("a") as a:
            for tf_not_exist_genkei in tf_not_exist_genkeis:
                a.write("{}\t{}\n".format(tf_not_exist_genkei,0))

    logger.info("処理が完了しました")

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--wikipedia_data_root_dirname",type=str,default="../../Data/Wikipedia")
    parser.add_argument("--idf_filepath",type=str,default="../../Data/idf.txt")
    parser.add_argument("--start_index",type=int,default=0)
    parser.add_argument("--end_index",type=int)
    args=parser.parse_args()

    main(args)
