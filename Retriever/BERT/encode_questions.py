import argparse
import json
import logging
import torch
from pathlib import Path
from transformers import AutoTokenizer,BertConfig,BertModel
from tqdm import tqdm
from typing import List,Tuple

logging_fmt="%(asctime)s %(levelname)s: %(message)s"
logging.basicConfig(format=logging_fmt)
logger=logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_samples(samples_filepath:str)->Tuple[List[str],List[str]]:
    qids:List[str]=[]
    questions:List[str]=[]

    with open(samples_filepath,"r") as r:
        lines=r.read().splitlines()

    for line in lines:
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

def main(args):
    logger.info(args)

    samples_filepath:str=args.samples_filepath
    bert_model_name:str=args.bert_model_name
    question_vectors_save_dirname:str=args.question_vectors_save_dirname

    logger.info("処理を行う準備をしています...")

    question_vectors_save_dir=Path(question_vectors_save_dirname)
    question_vectors_save_dir.mkdir(parents=True,exist_ok=True)

    qids,questions=load_samples(samples_filepath)

    config=BertConfig.from_pretrained(bert_model_name)
    bert=BertModel.from_pretrained(bert_model_name)
    tokenizer=AutoTokenizer.from_pretrained(bert_model_name)

    bert.eval()
    bert.to(device)

    logger.info("処理を行っています...")

    for qid,question in tqdm(zip(qids,questions),total=len(qids)):
        question_vector=get_question_vector(question,bert,tokenizer,config.max_length)

        save_file=question_vectors_save_dir.joinpath("{}.pt".format(qid))
        torch.save(question_vector.cpu(),save_file)

    logger.info("処理が完了しました")

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--samples_filepath",type=str,default="../../Data/aio_02_train.jsonl")
    parser.add_argument("--bert_model_name",type=str,default="cl-tohoku/bert-base-japanese-whole-word-masking")
    parser.add_argument("--question_vectors_save_dirname",type=str,default="../../Data/QuestionVector")
    args=parser.parse_args()

    main(args)
