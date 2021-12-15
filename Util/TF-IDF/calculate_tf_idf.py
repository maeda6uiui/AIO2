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

    logger.info("TF-IDFを計算する準備をしています...")

    wikipedia_data_root_dir=Path(wikipedia_data_root_dirname)
    wikipedia_data_dirs=wikipedia_data_root_dir.glob("*")
    wikipedia_data_dirs=list(wikipedia_data_dirs)
    wikipedia_data_dirs.sort()

    if end_index is None:
        end_index=len(wikipedia_data_dirs)

    logger.info("{}から{}のWikipedia記事に対して処理を行います".format(start_index,end_index))

    logger.info("IDF値を読み込んでいます...")

    genkeis=[]
    idfs=[]
    with open(idf_filepath,"r") as r:
        for line in r:
            if line=="":
                continue

            genkei,idf=line.split("\t")
            idf=float(idf)

            genkeis.append(genkei)
            idfs.append(idf)

    logger.info("TF-IDFの計算を行っています...")

    for idx,wikipedia_data_dir in enumerate(tqdm(wikipedia_data_dirs)):
        if idx<start_index:
            continue
        if idx>=end_index:
            break

        tf_file:Path=wikipedia_data_dir.joinpath("tf.txt")
        with tf_file.open("r") as r:
            lines=r.read().splitlines()

        #tf.txtとidf.txtは単語が同じ順番で整列されているとする
        tf_idfs=[]
        for i,line in enumerate(lines):
            tf=line.split("\t")[1]
            tf=float(tf)

            idf=idfs[i]
            tf_idfs.append(tf*idf)

        tf_idf_file:Path=wikipedia_data_dir.joinpath("tf_idf.txt")
        with tf_idf_file.open("w") as w:
            for genkei,tf_idf in zip(genkeis,tf_idfs):
                w.write("{}\t{}\n".format(genkei,tf_idf))

    logger.info("処理が完了しました")

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--wikipedia_data_root_dirname",type=str,default="../../Data/Wikipedia")
    parser.add_argument("--idf_filepath",type=str,default="../../Data/idf.txt")
    parser.add_argument("--start_index",type=int,default=0)
    parser.add_argument("--end_index",type=int)
    args=parser.parse_args()

    main(args)
