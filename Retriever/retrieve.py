import argparse
import json
import logging
import torch
from pathlib import Path
from transformers import AutoTokenizer,BertConfig,BertModel
from tqdm import tqdm
from typing import List,Tuple

import sys
sys.path.append(".")

from models import RelevanceScoreCalculator

logging_fmt="%(asctime)s %(levelname)s: %(message)s"
logging.basicConfig(format=logging_fmt)
logger=logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_questions(samples_filepath:str)->Tuple[List[str],List[str]]:
    qids:List[str]=[]
    questions:List[str]=[]

    with open(samples_filepath,"r") as r:
        for line in r:
            if line=="":
                continue

            data=json.loads(line)
            qid=data["qid"]
            question=data["question"]

            qids.append(qid)
            questions.append(question)

    return qids,questions

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

def retrieve(
    question:str,
    wikipedia_data_dirs:List[Path],
    score_calculator:RelevanceScoreCalculator,
    bert:BertModel,
    tokenizer:AutoTokenizer,
    max_length:int,
    k:int):
    question_vector=get_question_vector(question,bert,tokenizer,max_length)

    num_wikipedia_articles=len(wikipedia_data_dirs)
    scores=torch.empty(num_wikipedia_articles,device=device)

    for idx,wikipedia_data_dir in enumerate(wikipedia_data_dirs):
        document_vector=load_document_vector(wikipedia_data_dir)

        with torch.no_grad():
            score=score_calculator(question_vector,document_vector)
            score=torch.squeeze(score)

            scores[idx]=score

    top_k_scores,top_k_indices=torch.topk(scores,k=k)

    top_k_titles:List[str]=[]
    top_k_title_hashes:List[str]=[]

    for i in range(k):
        wikipedia_data_dir:Path=wikipedia_data_dirs[top_k_indices[i].item()]
        title_hash=wikipedia_data_dir.name

        title_file=wikipedia_data_dir.joinpath("title.txt")
        with title_file.open("r") as r:
            title=r.read().splitlines[0]

        top_k_titles.append(title)
        top_k_title_hashes.append(title_hash)

    ret={
        "top_k_scores":top_k_scores.cpu(),
        "top_k_titles":top_k_titles,
        "top_k_title_hashes":top_k_title_hashes
    }
    return ret

def main(args):
    logger.info(args)

    samples_filepath:str=args.samples_filepath
    wikipedia_data_root_dirname:str=args.wikipedia_data_root_dirname
    results_save_filepath:str=args.results_save_filepath
    bert_model_name:str=args.bert_model_name
    score_calculator_filepath:str=args.score_calculator_filepath
    k:int=args.k

    logger.info("関連度スコア計算の準備を行っています...")

    wikipedia_data_root_dir=Path(wikipedia_data_root_dirname)
    wikipedia_data_dirs=wikipedia_data_root_dir.glob("*")
    wikipedia_data_dirs=list(wikipedia_data_dirs)

    qids,questions=load_questions(samples_filepath)

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

    logger.info("関連度スコアの計算を行っています...")

    with open(results_save_filepath,"w") as w:
        for qid,question in tqdm(zip(qids,questions)):
            retrieval_results=retrieve(
                question,
                wikipedia_data_dirs,
                score_calculator,
                bert,
                tokenizer,
                config.max_length,
                k)
            
            top_k_titles=retrieval_results["top_k_titles"]
            top_k_title_hashes=retrieval_results["top_k_title_hashes"]
            top_k_scores=retrieval_results["top_k_scores"].tolist()

            output={
                "qid":qid,
                "top_k_titles":top_k_titles,
                "top_k_title_hashes":top_k_title_hashes,
                "top_k_scores":top_k_scores
            }

            line=json.dumps(output,ensure_ascii=False)
            w.write("{}\n".format(line))

    logger.info("処理が完了しました")

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--samples_filepath",type=str,default="../Data/aio_02_train.jsonl")
    parser.add_argument("--wikipedia_data_root_dirname",type=str,default="../Data/Wikipedia")
    parser.add_argument("--results_save_filepath",type=str,default="../Data/Retriever/train_top_ks.jsonl")
    parser.add_argument("--bert_model_name",type=str,default="cl-tohoku/bert-base-japanese-whole-word-masking")
    parser.add_argument("--score_calculator_filepath",type=str,default="../Data/Retriever/score_calculator.pt")
    parser.add_argument("--k",type=int,default=10)
    args=parser.parse_args()

    main(args)
