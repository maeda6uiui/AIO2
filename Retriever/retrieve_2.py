import argparse
import json
import logging
import numpy as np
import torch
from pathlib import Path
from transformers import BertConfig
from scipy import sparse
from tqdm import tqdm
from typing import Dict,List,Tuple

import sys
sys.path.append("./BERT")
sys.path.append("./TF-IDF")

from models import RelevanceScoreCalculator
from tf_idf import GenkeiExtractor,TFIDFCalculator

logging_fmt="%(asctime)s %(levelname)s: %(message)s"
logging.basicConfig(format=logging_fmt)
logger=logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def convert_csr_to_tensor(csr:sparse.csr_matrix)->torch.FloatTensor:
    coo=csr.tocoo()

    values=coo.data
    indices=np.vstack((coo.row,coo.col))

    i=torch.LongTensor(indices)
    v=torch.FloatTensor(values)
    shape=coo.shape

    tensor=torch.sparse_coo_tensor(i,v,torch.Size(shape))

    return tensor

def load_corpus_tensors(corpus_tensors_dir:Path)->List[torch.FloatTensor]:
    info={}
    info_file=corpus_tensors_dir.joinpath("info.txt")
    with info_file.open("r") as r:
        for line in r:
            key,value=line.split("=")
            info[key]=value

    num_divisions=int(info["num_divisions"])

    corpus_tensors=[]
    for i in range(num_divisions):
        indices_file=corpus_tensors_dir.joinpath("indices_{}.pt".format(i))
        values_file=corpus_tensors_dir.joinpath("values_{}.pt".format(i))
        size_file=corpus_tensors_dir.joinpath("size_{}.pt".format(i))

        indices=torch.load(indices_file,map_location=torch.device("cpu"))
        values=torch.load(values_file,map_location=torch.device("cpu"))
        size=torch.load(size_file,map_location=torch.device("cpu"))

        corpus_tensor=torch.sparse_coo_tensor(indices,values,size,device=device)
        corpus_tensors.append(corpus_tensor)

    return corpus_tensors

def load_questions(
    samples_filepath:str,
    start_index:int,
    end_index:int)->Tuple[List[str],List[str],List[List[str]]]:
    qids:List[str]=[]
    questions:List[str]=[]
    answers:List[str]=[]

    with open(samples_filepath,"r") as r:
        lines=r.read().splitlines()

    if end_index is None:
        end_index=len(lines)

    lines=lines[start_index:end_index]

    for line in lines:
        data=json.loads(line)
        qid=data["qid"]
        question=data["question"]
        this_answers=data["answers"]

        qids.append(qid)
        questions.append(question)
        answers.append(this_answers)

    return qids,questions,answers

def load_question_vectors(question_vector_files:List[Path])->Dict[str,torch.FloatTensor]:
    question_vectors:Dict[str,torch.FloatTensor]={}

    for question_vector_file in tqdm(question_vector_files):
        qid=question_vector_file.stem
        question_vector=torch.load(question_vector_file,map_location=torch.device("cpu"))

        question_vectors[qid]=question_vector.to(device)

    return question_vectors

def load_document_vector(wikipedia_data_dir:Path)->torch.FloatTensor:
    document_vector_file=wikipedia_data_dir.joinpath("vector.pt")
    document_vector=torch.load(document_vector_file,map_location=torch.device("cpu"))

    return document_vector.to(device)

def load_document_vectors(wikipedia_data_dirs:List[Path],dim_document_vector:int)->torch.FloatTensor:
    num_wikipedia_articles=len(wikipedia_data_dirs)
    document_vectors=torch.empty(num_wikipedia_articles,dim_document_vector)

    for idx,wikipedia_data_dir in enumerate(tqdm(wikipedia_data_dirs)):
        document_vector_file=wikipedia_data_dir.joinpath("vector.pt")
        document_vector=torch.load(document_vector_file,map_location=torch.device("cpu"))
        document_vectors[idx]=torch.squeeze(document_vector)

    return document_vectors.to(device)

def retrieve_by_score_calculator(
    question_vector:torch.FloatTensor,
    document_vectors:torch.FloatTensor,
    score_calculator:RelevanceScoreCalculator,
    batch_size:int)->np.ndarray:
    num_wikipedia_articles=document_vectors.size(0)
    all_scores=torch.empty(0,1,device=device)

    for i in range(0,num_wikipedia_articles,batch_size):
        this_document_vectors=document_vectors[i:i+batch_size]
        question_vectors=question_vector.expand(this_document_vectors.size(0),-1)

        with torch.no_grad():
            scores=score_calculator(question_vectors,this_document_vectors) #(N, 1)
            all_scores=torch.cat([all_scores,scores],dim=0)

    all_scores=torch.squeeze(all_scores)
    
    return all_scores

def retrieve_by_tf_idf(
    question:str,
    genkei_extractor:GenkeiExtractor,
    tf_idf_calculator:TFIDFCalculator,
    corpus_tensors:List[torch.FloatTensor])->np.ndarray:
    q_genkeis=genkei_extractor.extract_genkeis_from_text(question)
    q_genkeis=" ".join(q_genkeis)

    q_vector=tf_idf_calculator.transform_query(q_genkeis)
    q_vector=convert_csr_to_tensor(q_vector).to_dense()
    q_vector=torch.squeeze(q_vector)
    q_vector=q_vector.to(device)

    cosine_similarities=torch.empty(0,dtype=torch.float,device=device)

    for corpus_tensor in corpus_tensors:
        cs_temp=torch.mv(corpus_tensor,q_vector)
        cosine_similarities=torch.cat([cosine_similarities,cs_temp],dim=0)

    return cosine_similarities

def main(args):
    logger.info(args)

    #limit_num_wikipedia_dataを設定すると関連度スコアの乗算にてエラーが発生する
    #このオプションはそれ以前の処理が正しく動作しているか確認する場合に設定する
    #(Wikipediaの文書ベクトルをすべて読み込むのは時間がかかるため)

    samples_filepath:str=args.samples_filepath
    start_index:int=args.start_index
    end_index:int=args.end_index
    wikipedia_data_root_dirname:str=args.wikipedia_data_root_dirname
    question_vectors_dirname:str=args.question_vectors_dirname
    limit_num_wikipedia_data:int=args.limit_num_wikipedia_data
    results_save_filepath:str=args.results_save_filepath
    bert_model_name:str=args.bert_model_name
    score_calculator_filepath:str=args.score_calculator_filepath
    mecab_dictionary_dirname:str=args.mecab_dictionary_dirname
    corpus_tensors_dirname:str=args.corpus_tensors_dirname
    vectorizer_filepath:str=args.vectorizer_filepath
    batch_size:int=args.batch_size
    k:int=args.k

    logger.info("関連度スコア計算の準備を行っています...")

    wikipedia_data_root_dir=Path(wikipedia_data_root_dirname)
    wikipedia_data_dirs=wikipedia_data_root_dir.glob("*")
    wikipedia_data_dirs=list(wikipedia_data_dirs)
    wikipedia_data_dirs.sort()

    if limit_num_wikipedia_data is not None:
        wikipedia_data_dirs=wikipedia_data_dirs[:limit_num_wikipedia_data]

    qids,questions,answers=load_questions(samples_filepath,start_index,end_index)

    config=BertConfig.from_pretrained(bert_model_name)
    score_calculator=RelevanceScoreCalculator(config.hidden_size)
    state_dict=torch.load(score_calculator_filepath,map_location=torch.device("cpu"))
    score_calculator.load_state_dict(state_dict)

    score_calculator.eval()
    score_calculator.to(device)

    genkei_extractor=GenkeiExtractor(mecab_dictionary_dirname,1024*1024)

    corpus_tensors_dir=Path(corpus_tensors_dirname)
    corpus_tensors=load_corpus_tensors(corpus_tensors_dir)

    tf_idf_calculator=TFIDFCalculator()
    tf_idf_calculator.load_vectorizer(vectorizer_filepath)

    logger.info("問題文の特徴量ベクトルを読み込んでいます...")

    question_vectors_dir=Path(question_vectors_dirname)
    question_vector_files=question_vectors_dir.glob("*")
    question_vector_files=list(question_vector_files)
    question_vector_files.sort()

    question_vectors=load_question_vectors(question_vector_files)

    logger.info("Wikipedia記事の特徴量ベクトルを読み込んでいます...")

    document_vectors=load_document_vectors(wikipedia_data_dirs,config.hidden_size)

    logger.info("関連度スコアの計算を行っています...")

    with open(results_save_filepath,"a") as w:
        for qid,question,this_answers in tqdm(zip(qids,questions,answers),total=len(qids)):
            question_vector=question_vectors[qid]

            calculator_scores=retrieve_by_score_calculator(
                question_vector,
                document_vectors,
                score_calculator,
                batch_size
            )

            tf_idf_scores=retrieve_by_tf_idf(
                question,
                genkei_extractor,
                tf_idf_calculator,
                corpus_tensors
            )

            scores=torch.mul(calculator_scores,tf_idf_scores)
            scores=scores.cpu().detach().numpy()

            top_k_indices=np.argpartition(scores,-k)[-k:]
            top_k_indices=top_k_indices[np.argsort(-scores[top_k_indices])]

            top_k_scores=scores[top_k_indices]
            top_k_scores=top_k_scores.tolist()

            top_k_titles=[]
            for i in range(k):
                wikipedia_data_dir=wikipedia_data_dirs[top_k_indices[i]]

                title_file=wikipedia_data_dir.joinpath("title.txt")
                with title_file.open("r") as r:
                    title=r.read().splitlines()[0]
                    top_k_titles.append(title)

            output={
                "qid":qid,
                "question":question,
                "answers":this_answers,
                "top_k_titles":top_k_titles,
                "top_k_scores":top_k_scores
            }

            line=json.dumps(output,ensure_ascii=False)
            w.write("{}\n".format(line))

    logger.info("処理が完了しました")

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--samples_filepath",type=str,default="../Data/aio_02_train.jsonl")
    parser.add_argument("--start_index",type=int,default=0)
    parser.add_argument("--end_index",type=int)
    parser.add_argument("--wikipedia_data_root_dirname",type=str,default="../Data/Wikipedia")
    parser.add_argument("--question_vectors_dirname",type=str,default="../Data/QuestionVector")
    parser.add_argument("--limit_num_wikipedia_data",type=int)
    parser.add_argument("--results_save_filepath",type=str,default="../Data/Retriever/train_top_ks.jsonl")
    parser.add_argument("--bert_model_name",type=str,default="cl-tohoku/bert-base-japanese-whole-word-masking")
    parser.add_argument("--score_calculator_filepath",type=str,default="../Data/Retriever/BERT/score_calculator.pt")
    parser.add_argument("--mecab_dictionary_dirname",type=str,default="/usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd")
    parser.add_argument("--corpus_tensors_dirname",type=str,default="../Data/Retriever/TF-IDF/CorpusTensor")
    parser.add_argument("--vectorizer_filepath",type=str,default="../Data/Retriever/TF-IDF/vectorizer.pkl")
    parser.add_argument("--batch_size",type=int,default=512)
    parser.add_argument("--k",type=int,default=100)
    args=parser.parse_args()

    main(args)
