import argparse
import logging
import numpy as np
from pathlib import Path
from scipy import sparse
from sklearn.metrics.pairwise import linear_kernel
from tqdm import tqdm

import sys
sys.path.append(".")
from tf_idf import GenkeiExtractor,TFIDFCalculator

logging_fmt="%(asctime)s %(levelname)s: %(message)s"
logging.basicConfig(format=logging_fmt)
logger=logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

def main(args):
    logger.info(args)

    input_text:str=args.input_text
    wikipedia_data_root_dirname:str=args.wikipedia_data_root_dirname
    mecab_dictionary_dirname:str=args.mecab_dictionary_dirname
    corpus_matrix_filepath:str=args.corpus_matrix_filepath
    vectorizer_filepath:str=args.vectorizer_filepath
    output_filepath:str=args.output_filepath

    logger.info("処理を行う準備をしています...")

    wikipedia_data_root_dir=Path(wikipedia_data_root_dirname)
    wikipedia_data_dirs=wikipedia_data_root_dir.glob("*")
    wikipedia_data_dirs=list(wikipedia_data_dirs)
    wikipedia_data_dirs.sort()

    genkei_extractor=GenkeiExtractor(mecab_dictionary_dirname,1024*1024)

    corpus_matrix=sparse.load_npz(corpus_matrix_filepath)

    tf_idf_calculator=TFIDFCalculator()
    tf_idf_calculator.load_vectorizer(vectorizer_filepath)

    logger.info("関連度の計算を行っています...")

    q_genkeis=genkei_extractor.extract_genkeis_from_text(input_text)
    q_genkeis=" ".join(q_genkeis)

    q_vector=tf_idf_calculator.transform_query(q_genkeis)

    cosine_similarities=linear_kernel(corpus_matrix,q_vector).flatten()

    indices=np.argsort(-cosine_similarities)
    scores=cosine_similarities[indices]

    logger.info("結果をファイルに出力しています...")

    with open(output_filepath,"w") as w:
        for i in tqdm(range(indices.shape[0])):
            index=indices[i]
            score=scores[i]

            wikipedia_data_dir=wikipedia_data_dirs[index]

            title_file=wikipedia_data_dir.joinpath("title.txt")
            with title_file.open("r") as r:
                title=r.read().splitlines()[0]

            w.write("{}\t{}\n".format(title,score))

    logger.info("処理が完了しました")

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--input_text",type=str)
    parser.add_argument("--wikipedia_data_root_dirname",type=str,default="../../Data/Wikipedia")
    parser.add_argument("--samples_filepath",type=str,default="../../Data/aio_02_train.jsonl")
    parser.add_argument("--mecab_dictionary_dirname",type=str,default="/usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd")
    parser.add_argument("--corpus_matrix_filepath",type=str,default="../../Data/Retriever/TF-IDF/corpus_matrix.npz")
    parser.add_argument("--vectorizer_filepath",type=str,default="../../Data/Retriever/TF-IDF/vectorizer.pkl")
    parser.add_argument("--output_filepath",type=str,default="../../Data/Retriever/TF-IDF/relevance_scores.txt")
    args=parser.parse_args()

    main(args)
