import argparse
import hashlib
import json
import logging
import transformers
from pathlib import Path
from transformers import AutoTokenizer,AutoConfig
from tqdm import tqdm
from typing import List,Tuple

logging_fmt="%(asctime)s %(levelname)s: %(message)s"
logging.basicConfig(format=logging_fmt)
logger=logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

transformers.logging.set_verbosity_error()

def get_md5_hash(text:str)->str:
    return hashlib.md5(text.encode()).hexdigest()

def load_retrieval_results(
    retrieval_results_filepath:str,
    limit_num_titles:int)->Tuple[List[str],List[str],List[List[str]],List[List[str]]]:
    qids=[]
    questions=[]
    answers=[]
    top_k_titles=[]

    with open(retrieval_results_filepath,"r") as r:
        for line in r:
            data=json.loads(line)

            qid=data["qid"]
            question=data["question"]
            this_answers=data["answers"]
            this_top_k_titles=data["top_k_titles"][:limit_num_titles]

            qids.append(qid)
            questions.append(question)
            answers.append(this_answers)
            top_k_titles.append(this_top_k_titles)

    return qids,questions,answers,top_k_titles

def get_answer_indices_from_context(
    question:str,
    answer:str,
    context:str,
    tokenizer:AutoTokenizer,
    tokenizer_max_length:int,
    context_max_length:int)->Tuple[List[int],List[int]]:
    context=context[:context_max_length]

    input_ids=tokenizer.encode(
        question,
        context,
        padding="max_length",
        max_length=tokenizer_max_length,
        truncation="only_second"
    )
    
    answer_tokens=tokenizer.tokenize(answer)
    answer_token_ids=tokenizer.convert_tokens_to_ids(answer_tokens)

    start_indices:List[int]=[]
    end_indices:List[int]=[]

    answer_length=len(answer_token_ids)

    for i in range(tokenizer_max_length):
        if input_ids[i]==answer_token_ids[0]:
            probable_window=input_ids[i:i+answer_length]
            if probable_window==answer_token_ids:
                start_index=i
                end_index=i+answer_length-1

                start_indices.append(start_index)
                end_indices.append(end_index)

    return start_indices,end_indices

def main(args):
    logger.info(args)

    retrieval_results_filepath:str=args.retrieval_results_filepath
    wikipedia_data_root_dirname:str=args.wikipedia_data_root_dirname
    bert_model_name:str=args.bert_model_name
    positive_output_filepath:str=args.positive_output_filepath
    negative_output_filepath:str=args.negative_output_filepath
    context_max_length:int=args.context_max_length
    limit_num_titles:int=args.limit_num_titles

    logger.info("処理を行う準備をしています...")

    wikipedia_data_root_dir=Path(wikipedia_data_root_dirname)

    qids,questions,answers,top_k_titles=load_retrieval_results(retrieval_results_filepath,limit_num_titles)

    config=AutoConfig.from_pretrained(bert_model_name)
    tokenizer=AutoTokenizer.from_pretrained(bert_model_name)

    logger.info("データを作成しています...")

    positives=[]
    negatives=[]

    for qid,question,this_answers,this_top_k_titles in tqdm(zip(qids,questions,answers,top_k_titles),total=len(qids)):
        for title in this_top_k_titles:
            title_hash=get_md5_hash(title)
            text_file=wikipedia_data_root_dir.joinpath(title_hash,"text.txt")
            with text_file.open("r") as r:
                context=r.read()

            answer_contained=False
            for answer in this_answers:
                start_indices,end_indices=get_answer_indices_from_context(
                    question,
                    answer,
                    context,
                    tokenizer,
                    config.max_position_embeddings,
                    context_max_length
                )
                if len(start_indices)==0:
                    continue
                
                ranges:List[str]=[]
                for start_index,end_index in zip(start_indices,end_indices):
                    range="{}-{}".format(start_index,end_index)
                    ranges.append(range)

                data={
                    "qid":qid,
                    "question":question,
                    "answer":answer,
                    "article_title":title,
                    "answer_ranges":ranges
                }
                positives.append(data)
                answer_contained=True

            if answer_contained==False:
                data={
                    "qid":qid,
                    "question":question,
                    "answer":this_answers[0],
                    "article_title":title,
                    "answer_ranges":["0-0"]
                }
                negatives.append(data)

    logger.info("作成したデータをファイルに書き込んでいます...")

    with open(positive_output_filepath,"w") as w:
        for positive in positives:
            line=json.dumps(positive,ensure_ascii=False)
            w.write(line)
            w.write("\n")

    with open(negative_output_filepath,"w") as w:
        for negative in negatives:
            line=json.dumps(negative,ensure_ascii=False)
            w.write(line)
            w.write("\n")

    logger.info("処理が完了しました")

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--retrieval_results_filepath",type=str,default="../Data/Retriever/train_top_ks.jsonl")
    parser.add_argument("--wikipedia_data_root_dirname",type=str,default="../Data/Wikipedia")
    parser.add_argument("--bert_model_name",type=str,default="cl-tohoku/bert-base-japanese-whole-word-masking")
    parser.add_argument("--positive_output_filepath",type=str,default="../Data/Reader/train_positive_samples.jsonl")
    parser.add_argument("--negative_output_filepath",type=str,default="../Data/Reader/train_negative_samples.jsonl")
    parser.add_argument("--context_max_length",type=int,default=3000)
    parser.add_argument("--limit_num_titles",type=int,default=100)
    args=parser.parse_args()

    main(args)
