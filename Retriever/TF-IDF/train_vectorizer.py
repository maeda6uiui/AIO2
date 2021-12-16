import argparse
import logging
from pathlib import Path

import sys
sys.path.append(".")
from tf_idf import TFIDFCalculator

logging_fmt="%(asctime)s %(levelname)s: %(message)s"
logging.basicConfig(format=logging_fmt)
logger=logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

def main(args):
    logger.info(args)

    wikipedia_data_root_dirname:str=args.wikipedia_data_root_dirname
    corpus_matrix_save_filepath:str=args.corpus_matrix_save_filepath
    vectorizer_save_filepath:str=args.vectorizer_save_filepath
    start_index:int=args.start_index
    end_index:int=args.end_index

    logger.info("処理を行う準備をしています...")

    wikipedia_data_root_dir=Path(wikipedia_data_root_dirname)
    wikipedia_data_dirs=wikipedia_data_root_dir.glob("*")
    wikipedia_data_dirs=list(wikipedia_data_dirs)
    wikipedia_data_dirs.sort()

    logger.info("Wikipedia記事の数: {}".format(len(wikipedia_data_dirs)))

    if end_index is None:
        end_index=len(wikipedia_data_dirs)

    logger.info("{}から{}のWikipediaデータを使用してVectorizerの学習を行います".format(start_index,end_index))

    wikipedia_data_dirs=wikipedia_data_dirs[start_index:end_index]

    tf_idf_calculator=TFIDFCalculator()

    logger.info("Vectorizerの学習を行っています...")

    tf_idf_calculator.fit_transform_corpus(wikipedia_data_dirs,corpus_matrix_save_filepath)
    tf_idf_calculator.save_vectorizer(vectorizer_save_filepath)

    logger.info("Vectorizerの学習が完了しました")

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--wikipedia_data_root_dirname",type=str,default="../../Data/Wikipedia")
    parser.add_argument("--corpus_matrix_save_filepath",type=str,default="../../Data/Retriever/TF-IDF/corpus_matrix.npz")
    parser.add_argument("--vectorizer_save_filepath",type=str,default="../../Data/Retriever/TF-IDF/vectorizer.pkl")
    parser.add_argument("--start_index",type=int,default=0)
    parser.add_argument("--end_index",type=int)
    args=parser.parse_args()

    main(args)
