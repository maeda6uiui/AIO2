import argparse
import hashlib
import json
import logging
import subprocess
import torch
from torch.utils.data import Dataset,DataLoader
from pathlib import Path
from transformers import AutoConfig,AutoTokenizer,BertForQuestionAnswering
from tqdm import tqdm
from typing import List

import sys
sys.path.append(".")
from models import Reader

logging_fmt="%(asctime)s %(levelname)s: %(message)s"
logging.basicConfig(format=logging_fmt)
logger=logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReaderEvalDataset(Dataset):
    def __init__(self):
        self.qids:List[str]=[]
        self.questions:List[str]=[]
        self.answers:List[List[str]]=[]
        self.top_k_titles:List[List[str]]=[]
        self.top_k_scores:List[List[float]]=[]

    def __len__(self):
        return len(self.qids)

    def __getitem__(self,idx:int):
        qid=self.qids[idx]
        question=self.questions[idx]
        this_answers=self.answers[idx]
        this_top_k_titles=self.top_k_titles[idx]
        this_top_k_scores=self.top_k_scores[idx]

        return qid,question,this_answers,this_top_k_titles,this_top_k_scores

    def append(
        self,
        qid:str,
        question:str,
        this_answers:List[List[str]],
        this_top_k_titles:List[str],
        this_top_k_scores:List[float]):
        self.qids.append(qid)
        self.questions.append(question)
        self.answers.append(this_answers)
        self.top_k_titles.append(this_top_k_titles)
        self.top_k_scores.append(this_top_k_scores)

def collate_fn_for_eval_dataset(batch):
    qids,questions,src_answers,src_top_k_titles,src_top_k_scores=list(zip(*batch))

    answers:List[str]=[]
    top_k_titles:List[str]=[]
    top_k_scores:List[float]=[]

    for i in range(len(src_answers[0])):
        answers.append(src_answers[0][i])
        
    for i in range(len(src_top_k_titles[0])):
        top_k_titles.append(src_top_k_titles[0][i])
        top_k_scores.append(src_top_k_scores[0][i])

    ret={
        "qid":qids[0],
        "question":questions[0],
        "answers":answers,
        "top_k_titles":top_k_titles,
        "top_k_scores":top_k_scores
    }
    return ret

def get_md5_hash(text:str)->str:
    return hashlib.md5(text.encode()).hexdigest()

def create_eval_dataset(samples_filepath:str,limit_num_samples:int=None)->ReaderEvalDataset:
    dataset=ReaderEvalDataset()

    with open(samples_filepath,"r") as r:
        lines=r.read().splitlines()

    if limit_num_samples is not None:
        lines=lines[:limit_num_samples]

    for line in lines:
        data=json.loads(line)

        qid=data["qid"]
        question=data["question"]
        answers=data["answers"]
        top_k_titles=data["top_k_titles"]
        top_k_scores=data["top_k_scores"]

        dataset.append(qid,question,answers,top_k_titles,top_k_scores)

    return dataset

def wakati_with_jumanpp(text:str)->str:
    p1=subprocess.Popen(["echo",text],stdout=subprocess.PIPE)
    p2=subprocess.Popen(["jumanpp"],stdin=p1.stdout,stdout=subprocess.PIPE)

    p1.stdout.close()

    output=p2.communicate()[0]
    lines=output.decode("utf8").splitlines()
    lines.pop()

    wakatis=[]
    for line in lines:
        wakati=line.split(" ")[0]
        wakatis.append(wakati)

    return " ".join(wakatis)

def create_eval_model_inputs(
    tokenizer:AutoTokenizer,
    max_length:int,
    wikipedia_data_root_dir:Path,
    question:str,
    top_k_titles:List[str],
    context_max_length:int):
    contexts:List[str]=[]
    for article_title in top_k_titles:
        title_hash=get_md5_hash(article_title)
        text_file=wikipedia_data_root_dir.joinpath(title_hash,"text.txt")

        with text_file.open("r") as r:
            context=r.read()
            context=context[:context_max_length]
            contexts.append(context)

    num_contexts=len(contexts)

    input_ids=torch.empty(num_contexts,max_length,dtype=torch.long)
    attention_mask=torch.empty(num_contexts,max_length,dtype=torch.long)
    token_type_ids=torch.empty(num_contexts,max_length,dtype=torch.long)

    question_wakati=wakati_with_jumanpp(question)

    for i in range(num_contexts):
        context_wakati=wakati_with_jumanpp(contexts[i])

        encode=tokenizer.encode_plus(
            question_wakati,
            context_wakati,
            padding="max_length",
            max_length=max_length,
            truncation="only_second",
            return_tensors="pt"
        )

        this_input_ids=torch.squeeze(encode["input_ids"])
        this_attention_mask=torch.squeeze(encode["attention_mask"])
        this_token_type_ids=torch.squeeze(encode["token_type_ids"])

        input_ids[i]=this_input_ids
        attention_mask[i]=this_attention_mask
        token_type_ids[i]=this_token_type_ids

    input_ids=input_ids.to(device)
    attention_mask=attention_mask.to(device)
    token_type_ids=token_type_ids.to(device)

    ret={
        "input_ids":input_ids,
        "attention_mask":attention_mask,
        "token_type_ids":token_type_ids
    }
    return ret

def eval(
    model:Reader,
    eval_dataloader:DataLoader,
    tokenizer:AutoTokenizer,
    max_length:int,
    wikipedia_data_root_dir:Path,
    eval_batch_size:int,
    context_max_length:int,
    limit_num_top_k:int):
    model.eval()

    question_count=0
    correct_count=0

    unk_id=tokenizer.convert_tokens_to_ids("[UNK]")

    qids:List[str]=[]
    questions:List[str]=[]
    answers:List[List[str]]=[]
    predicted_articles:List[str]=[]
    predicted_answers:List[str]=[]

    #問題1問を一つのバッチとするため、評価用データローダのバッチサイズは1で固定しておく
    #コマンドライン引数で設定するバッチサイズ(eval_batch_size)は、
    #一度にいくつの記事をモデルに入力するかを指定するためのもの
    for batch in tqdm(eval_dataloader):
        qid=batch["qid"]
        question=batch["question"]
        this_answers=batch["answers"]
        top_k_titles=batch["top_k_titles"]
        top_k_scores=batch["top_k_scores"]

        if limit_num_top_k is not None:
            top_k_titles=top_k_titles[:limit_num_top_k]
            top_k_scores=top_k_scores[:limit_num_top_k]

        inputs=create_eval_model_inputs(
            tokenizer,
            max_length,
            wikipedia_data_root_dir,
            question,
            top_k_titles,
            context_max_length
        )

        num_titles=len(top_k_titles)

        num_sub_batches=num_titles//eval_batch_size
        if num_titles%eval_batch_size!=0:
            num_sub_batches+=1

        start_logits=torch.empty(0,max_length)
        end_logits=torch.empty(0,max_length)
        plausibility_scores=torch.empty(0)

        for i in range(num_sub_batches):
            start_index=eval_batch_size*i
            end_index=eval_batch_size*(i+1)

            input_ids=inputs["input_ids"][start_index:end_index,:]
            attention_mask=inputs["attention_mask"][start_index:end_index,:]
            token_type_ids=inputs["token_type_ids"][start_index:end_index,:]

            sub_inputs={
                "input_ids":input_ids,
                "attention_mask":attention_mask,
                "token_type_ids":token_type_ids
            }

            with torch.no_grad():
                outputs=model(**sub_inputs)

                this_start_logits=outputs["start_logits"]
                this_end_logits=outputs["end_logits"]
                this_plausibility_scores=outputs["plausibility_scores"]

                this_start_logits=this_start_logits.cpu()
                this_end_logits=this_end_logits.cpu()
                this_plausibility_scores=this_plausibility_scores.cpu()

                this_start_logits=torch.softmax(this_start_logits,dim=1)
                this_end_logits=torch.softmax(this_end_logits,dim=1)

                start_logits=torch.cat([start_logits,this_start_logits],dim=0)
                end_logits=torch.cat([end_logits,this_end_logits],dim=0)

                plausibility_scores=torch.cat([plausibility_scores,this_plausibility_scores],dim=0)

        _,start_indices=torch.max(start_logits,dim=1)
        _,end_indices=torch.max(end_logits,dim=1)

        _,plausible_article_indices=torch.topk(plausibility_scores,k=num_titles,dim=0)

        plausible_article_exists=False
        plausible_article_index=-1
        answer_start_index=-1
        answer_end_index=-1
        for i in range(num_titles):
            plausible_article_index=plausible_article_indices[i].item()
            answer_start_index=start_indices[plausible_article_index].item()
            answer_end_index=end_indices[plausible_article_index].item()

            if answer_start_index==0 or answer_end_index==0:
                continue
            if answer_start_index>answer_end_index:
                continue
            if answer_end_index-answer_start_index>20:
                continue

            answer_ids=inputs["input_ids"][plausible_article_index,answer_start_index:answer_end_index+1].tolist()
            if unk_id in answer_ids:
                continue

            plausible_article_exists=True
            break

        predicted_answer="N/A"
        predicted_article="N/A"
        if plausible_article_exists:
            answer_ids=inputs["input_ids"][plausible_article_index,answer_start_index:answer_end_index+1].tolist()

            predicted_answer=tokenizer.decode(answer_ids)
            predicted_answer=predicted_answer.replace(" ","")

            if predicted_answer in this_answers:
                correct_count+=1

            predicted_article=top_k_titles[plausible_article_index]

        question_count+=1

        qids.append(qid)
        questions.append(question)
        answers.append(this_answers)
        predicted_articles.append(predicted_article)
        predicted_answers.append(predicted_answer)

    ret={
        "accuracy":correct_count/question_count*100,

        "qids":qids,
        "questions":questions,
        "answers":answers,
        "predicted_articles":predicted_articles,
        "predicted_answers":predicted_answers
    }
    return ret

def main(args):
    logger.info(args)

    eval_samples_filepath:str=args.eval_samples_filepath
    limit_num_eval_samples:int=args.limit_num_eval_samples
    reader_model_filepath:str=args.reader_model_filepath
    wikipedia_data_root_dirname:str=args.wikipedia_data_root_dirname
    model_name:str=args.model_name
    results_save_dirname:str=args.results_save_dirname
    eval_batch_size:int=args.eval_batch_size
    context_max_length:int=args.context_max_length
    limit_num_top_k:int=args.limit_num_top_k

    logger.info("モデルの評価を行う準備をしています...")

    results_save_dir=Path(results_save_dirname)
    results_save_dir.mkdir(parents=True,exist_ok=True)

    wikipedia_data_root_dir=Path(wikipedia_data_root_dirname)

    config=AutoConfig.from_pretrained(model_name)
    tokenizer=AutoTokenizer.from_pretrained(model_name)
    model=Reader(model_name)

    state_dict=torch.load(reader_model_filepath,map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.to(device)

    eval_dataset=create_eval_dataset(eval_samples_filepath,limit_num_samples=limit_num_eval_samples)
    eval_dataloader=DataLoader(eval_dataset,batch_size=1,shuffle=False,collate_fn=collate_fn_for_eval_dataset)

    logger.info("評価データの数: {}".format(len(eval_dataset)))

    logger.info("モデルの評価を開始します")

    eval_results=eval(
        model,
        eval_dataloader,
        tokenizer,
        config.max_position_embeddings,
        wikipedia_data_root_dir,
        eval_batch_size,
        context_max_length,
        limit_num_top_k
    )

    eval_accuracy=eval_results["accuracy"]

    logger.info("評価時の正解率(完全一致): {} %".format(eval_accuracy))

    eval_result_file=results_save_dir.joinpath("eval_result.txt")
    with eval_result_file.open("w") as w:
        w.write("評価時の正解率(完全一致): {} %\n".format(eval_accuracy))

    eval_predictions_file=results_save_dir.joinpath("eval_predictions.jsonl")
    with eval_predictions_file.open("w") as w:
        qids=eval_results["qids"]
        questions=eval_results["questions"]
        answers=eval_results["answers"]
        predicted_articles=eval_results["predicted_articles"]
        predicted_answers=eval_results["predicted_answers"]

        for qid,question,this_answers,predicted_article,predicted_answer in zip(qids,questions,answers,predicted_articles,predicted_answers):
            data={
                "qid":qid,
                "question":question,
                "answers":this_answers,
                "predicted_article":predicted_article,
                "predicted_answer":predicted_answer
            }
            line=json.dumps(data,ensure_ascii=False)

            w.write(line)
            w.write("\n")

    logger.info("モデルの評価が完了しました")

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--eval_samples_filepath",type=str,default="../../Data/Retriever/dev_top_ks.jsonl")
    parser.add_argument("--limit_num_eval_samples",type=int)
    parser.add_argument("--reader_model_filepath",type=str)
    parser.add_argument("--wikipedia_data_root_dirname",type=str,default="../../Data/Wikipedia")
    parser.add_argument("--model_name",type=str,default="nlp-waseda/roberta-base-japanese")
    parser.add_argument("--results_save_dirname",type=str,default="../../Data/Reader")
    parser.add_argument("--eval_batch_size",type=int,default=16)
    parser.add_argument("--context_max_length",type=int,default=3000)
    parser.add_argument("--limit_num_top_k",type=int)
    args=parser.parse_args()

    main(args)
