import argparse
import json
import logging
import numpy as np
import torch
from pathlib import Path
from scipy import sparse
from sklearn.metrics.pairwise import linear_kernel
from transformers import AutoTokenizer,BertConfig,BertModel
from tqdm import tqdm
from typing import List,Tuple

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
        if line=="":
            continue

        data=json.loads(line)
        qid=data["qid"]
        question=data["question"]
        this_answers=data["answers"]

        qids.append(qid)
        questions.append(question)
        answers.append(this_answers)

    return qids,questions,answers

def get_question_vector(
    question:str,
    bert:BertModel,
    tokenizer:AutoTokenizer,
    max_length:int)->torch.FloatTensor:
    q_inputs=tokenizer.encode_plus(
        question,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt")

    bert_inputs={
        "input_ids":q_inputs["input_ids"].to(device),
        "attention_mask":q_inputs["attention_mask"].to(device),
        "token_type_ids":q_inputs["token_type_ids"].to(device),
        "return_dict":True
    }

    with torch.no_grad():
        bert_outputs=bert(**bert_inputs)
        question_vector=bert_outputs["pooler_output"] #(1, hidden_size)

    return question_vector

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

def retrieve_by_bert(
    question:str,
    document_vectors:torch.FloatTensor,
    score_calculator:RelevanceScoreCalculator,
    bert:BertModel,
    tokenizer:AutoTokenizer,
    max_length:int,
    batch_size:int)->np.ndarray:
    question_vector=get_question_vector(question,bert,tokenizer,max_length) #(1, hidden_size)

    num_wikipedia_articles=document_vectors.size(0)
    all_scores=torch.empty(0,1,device=device)

    for i in range(0,num_wikipedia_articles,batch_size):
        this_document_vectors=document_vectors[i:i+batch_size]
        question_vectors=question_vector.expand(this_document_vectors.size(0),-1)

        with torch.no_grad():
            scores=score_calculator(question_vectors,this_document_vectors) #(N, 1)
            all_scores=torch.cat([all_scores,scores],dim=0)

    all_scores=torch.squeeze(all_scores)
    
    return all_scores.cpu().detach().numpy()

def retrieve_by_tf_idf(
    question:str,
    genkei_extractor:GenkeiExtractor,
    tf_idf_calculator:TFIDFCalculator,
    corpus_matrix)->np.ndarray:
    q_genkeis=genkei_extractor.extract_genkeis_from_text(question)
    q_genkeis=" ".join(q_genkeis)

    q_vector=tf_idf_calculator.transform_query(q_genkeis)

    cosine_similarities=linear_kernel(corpus_matrix,q_vector).flatten()

    return cosine_similarities

def main(args):
    logger.info(args)

    samples_filepath:str=args.samples_filepath
    start_index:int=args.start_index
    end_index:int=args.end_index
    wikipedia_data_root_dirname:str=args.wikipedia_data_root_dirname
    limit_num_wikipedia_data:int=args.limit_num_wikipedia_data
    results_save_filepath:str=args.results_save_filepath
    bert_model_name:str=args.bert_model_name
    score_calculator_filepath:str=args.score_calculator_filepath
    mecab_dictionary_dirname:str=args.mecab_dictionary_dirname
    corpus_matrix_filepath:str=args.corpus_matrix_filepath
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

    titles=[]
    for wikipedia_data_dir in wikipedia_data_dirs:
        title_file=wikipedia_data_dir.joinpath("title.txt")
        with title_file.open("r") as r:
            title=r.read().splitlines()[0]
            titles.append(title)

    qids,questions,answers=load_questions(samples_filepath,start_index,end_index)

    config=BertConfig.from_pretrained(bert_model_name)
    score_calculator=RelevanceScoreCalculator(config.hidden_size)
    state_dict=torch.load(score_calculator_filepath,map_location=torch.device("cpu"))
    score_calculator.load_state_dict(state_dict)

    bert=BertModel.from_pretrained(bert_model_name)
    tokenizer=AutoTokenizer.from_pretrained(bert_model_name)

    score_calculator.eval()
    score_calculator.to(device)
    bert.eval()
    bert.to(device)

    genkei_extractor=GenkeiExtractor(mecab_dictionary_dirname,1024*1024)

    corpus_matrix=sparse.load_npz(corpus_matrix_filepath)

    tf_idf_calculator=TFIDFCalculator()
    tf_idf_calculator.load_vectorizer(vectorizer_filepath)

    logger.info("Wikipedia記事の特徴量ベクトルを読み込んでいます...")

    document_vectors=load_document_vectors(wikipedia_data_dirs,config.hidden_size)

    logger.info("関連度スコアの計算を行っています...")

    with open(results_save_filepath,"w") as w:
        for qid,question,this_answers in tqdm(zip(qids,questions,answers),total=len(qids)):
            bert_scores=retrieve_by_bert(
                question,
                document_vectors,
                score_calculator,
                bert,
                tokenizer,
                config.max_length,
                batch_size)

            tf_idf_scores=retrieve_by_tf_idf(
                question,
                genkei_extractor,
                tf_idf_calculator,
                corpus_matrix)

            scores=bert_scores*tf_idf_scores

            top_k_indices=np.argpartition(scores,-k)[-k:]
            top_k_indices=top_k_indices[np.argsort(scores[top_k_indices])]

            top_k_scores=scores[top_k_indices]
            top_k_scores=top_k_scores.tolist()

            top_k_titles=[]
            for i in range(k):
                title=titles[top_k_indices[i]]
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
    parser.add_argument("--limit_num_wikipedia_data",type=int)
    parser.add_argument("--results_save_filepath",type=str,default="../Data/Retriever/train_top_ks.jsonl")
    parser.add_argument("--bert_model_name",type=str,default="cl-tohoku/bert-base-japanese-whole-word-masking")
    parser.add_argument("--score_calculator_filepath",type=str,default="../Data/Retriever/score_calculator.pt")
    parser.add_argument("--mecab_dictionary_dirname",type=str,default="/usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd")
    parser.add_argument("--corpus_matrix_filepath",type=str,default="../Data/Retriever/TF-IDF/corpus_matrix.npz")
    parser.add_argument("--vectorizer_filepath",type=str,default="../Data/Retriever/TF-IDF/vectorizer.pkl")
    parser.add_argument("--batch_size",type=int,default=512)
    parser.add_argument("--k",type=int,default=100)
    args=parser.parse_args()

    main(args)
