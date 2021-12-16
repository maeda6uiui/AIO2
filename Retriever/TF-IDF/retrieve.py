import argparse
import json
import logging
import numpy as np
from pathlib import Path
from scipy import sparse
from sklearn.metrics.pairwise import linear_kernel
from tqdm import tqdm
from typing import List,Tuple

import sys
sys.path.append(".")
from tf_idf import GenkeiExtractor,TFIDFCalculator

logging_fmt="%(asctime)s %(levelname)s: %(message)s"
logging.basicConfig(format=logging_fmt)
logger=logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

def load_questions(samples_filepath:str)->Tuple[List[str],List[str],List[List[str]]]:
    qids=[]
    questions=[]
    answers=[]

    with open(samples_filepath,"r") as r:
        for line in r:
            data=json.loads(line)

            qid=data["qid"]
            question=data["question"]
            this_answers=data["answers"]

            qids.append(qid)
            questions.append(question)
            answers.append(this_answers)

    return qids,questions,answers

def main(args):
    logger.info(args)

    wikipedia_data_root_dirname:str=args.wikipedia_data_root_dirname
    samples_filepath:str=args.samples_filepath
    mecab_dictionary_dirname:str=args.mecab_dictionary_dirname
    corpus_matrix_filepath:str=args.corpus_matrix_filepath
    vectorizer_filepath:str=args.vectorizer_filepath
    output_filepath:str=args.output_filepath
    k:int=args.k

    logger.info("処理を行う準備をしています...")

    wikipedia_data_root_dir=Path(wikipedia_data_root_dirname)
    wikipedia_data_dirs=wikipedia_data_root_dir.glob("*")
    wikipedia_data_dirs=list(wikipedia_data_dirs)
    wikipedia_data_dirs.sort()

    qids,questions,answers=load_questions(samples_filepath)

    genkei_extractor=GenkeiExtractor(mecab_dictionary_dirname,1024*1024)

    corpus_matrix=sparse.load_npz(corpus_matrix_filepath)

    tf_idf_calculator=TFIDFCalculator()
    tf_idf_calculator.load_vectorizer(vectorizer_filepath)

    logger.info("関連度の計算を行っています...")

    with open(output_filepath,"w") as w:
        for qid,question,this_answers in tqdm(zip(qids,questions,answers),total=len(qids)):
            q_genkeis=genkei_extractor.extract_genkeis_from_text(question)
            q_genkeis=" ".join(q_genkeis)

            q_vector=tf_idf_calculator.transform_query(q_genkeis)

            cosine_similarities=linear_kernel(corpus_matrix,q_vector).flatten()
            top_k_indices=np.argpartition(cosine_similarities,-k)[-k:]
            top_k_indices=top_k_indices[np.argsort(cosine_similarities[top_k_indices])]
            
            top_k_scores=cosine_similarities[top_k_indices]
            top_k_scores=top_k_scores.tolist()

            top_k_titles=[]
            for i in range(k):
                wikipedia_data_dir=wikipedia_data_dirs[top_k_indices[i]]

                title_file=wikipedia_data_dir.joinpath("title.txt")
                with title_file.open("r") as r:
                    title=r.read().splitlines()[0]
                    top_k_titles.append(title)

            data={
                "qid":qid,
                "question":question,
                "answers":this_answers,
                "top_k_titles":top_k_titles,
                "top_k_scores":top_k_scores
            }
            line=json.dumps(data,ensure_ascii=False)

            w.write(line)
            w.write("\n")

    logger.info("処理が完了しました")

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--wikipedia_data_root_dirname",type=str,default="../../Data/Wikipedia")
    parser.add_argument("--samples_filepath",type=str,default="../../Data/aio_02_train.jsonl")
    parser.add_argument("--mecab_dictionary_dirname",type=str,default="/usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd")
    parser.add_argument("--corpus_matrix_filepath",type=str,default="../../Data/Retriever/TF-IDF/corpus_matrix.npz")
    parser.add_argument("--vectorizer_filepath",type=str,default="../../Data/Retriever/TF-IDF/vectorizer.pkl")
    parser.add_argument("--output_filepath",type=str,default="../../Data/Retriever/TF-IDF/train_top_ks.jsonl")
    parser.add_argument("-k","--k",type=int,default=100)
    args=parser.parse_args()

    main(args)
