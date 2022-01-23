import argparse
import hashlib
import json
import logging
import re
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

def set_global_logging_level(level=logging.ERROR, prefices=[""]):
    """
    Override logging levels of different modules based on their name as a prefix.
    It needs to be invoked after the modules have been loaded so that their loggers have been initialized.

    Args:
        - level: desired level. e.g. logging.INFO. Optional. Default is logging.ERROR
        - prefices: list of one or more str prefices to match (e.g. ["transformers", "torch"]). Optional.
          Default is `[""]` to match all active loggers.
          The match is a case-sensitive `module_name.startswith(prefix)`
    """
    prefix_re = re.compile(fr'^(?:{ "|".join(prefices) })')
    for name in logging.root.manager.loggerDict:
        if re.match(prefix_re, name):
            logging.getLogger(name).setLevel(level)

set_global_logging_level(logging.ERROR,["transformers"])

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

    for i in range(num_contexts):
        encode=tokenizer.encode_plus(
            question,
            contexts[i],
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

def create_eval_model_inputs_2(
    tokenizer:AutoTokenizer,
    max_length:int,
    wikipedia_data_root_dir:Path,
    plausible_article_title:str,
    question:str):
    title_hash=get_md5_hash(plausible_article_title)
    context_file=wikipedia_data_root_dir.joinpath(title_hash,"text.txt")

    with context_file.open("r") as r:
        context=r.read()

    question_input_ids=tokenizer.encode(question,return_tensors="pt")[0]
    context_input_ids=tokenizer.encode(context,return_tensors="pt")[0]

    sep_id=tokenizer.convert_tokens_to_ids("[SEP]")
    sep_id=torch.tensor(sep_id,dtype=torch.long)
    sep_id=torch.unsqueeze(sep_id,0)

    len_question_input_ids=question_input_ids.size(0)
    len_context_input_ids=context_input_ids.size(0)

    len_context_chunk=max_length-len_question_input_ids-1
    num_chunks=len_context_input_ids//len_context_chunk
    if len_context_input_ids%len_context_chunk!=0:
        num_chunks+=1

    input_ids=torch.empty(num_chunks,max_length,dtype=torch.long)
    attention_mask=torch.empty(num_chunks,max_length,dtype=torch.long)

    token_type_ids=torch.ones(num_chunks,max_length,dtype=torch.long)
    token_type_ids[:,:len_question_input_ids]=0

    for i in range(num_chunks):
        context_input_ids_chunk=context_input_ids[i*len_context_chunk:(i+1)*len_context_chunk]
        context_input_ids_chunk=torch.cat([context_input_ids_chunk,sep_id],dim=0)

        input_ids_chunk=torch.cat([question_input_ids,context_input_ids_chunk],dim=0)
        attention_mask_chunk=torch.ones(max_length,dtype=torch.long)

        if i==num_chunks-1:
            len_zero_padding=max_length-input_ids_chunk.size(0)
            zero_padding=torch.zeros(len_zero_padding,dtype=torch.long)

            attention_mask_chunk[input_ids_chunk.size(0):]=0

            input_ids_chunk=torch.cat([input_ids_chunk,zero_padding],dim=0)

        input_ids[i]=input_ids_chunk
        attention_mask[i]=attention_mask_chunk

    input_ids=input_ids.to(device)
    attention_mask=attention_mask.to(device)
    token_type_ids=token_type_ids.to(device)

    ret={
        "num_chunks":num_chunks,
        "input_ids":input_ids,
        "attention_mask":attention_mask,
        "token_type_ids":token_type_ids
    }
    return ret

def get_most_plausible_article_index(
    model:Reader,
    tokenizer:AutoTokenizer,
    max_length:int,
    wikipedia_data_root_dir:Path,
    eval_batch_size:int,
    question:str,
    top_k_titles:List[str],
    context_max_length:int)->int:
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

            this_plausibility_scores=outputs["plausibility_scores"]
            this_plausibility_scores=this_plausibility_scores.cpu() #(1, eval_batch_size)
            this_plausibility_scores=torch.squeeze(this_plausibility_scores)    #(eval_batch_size)
            plausibility_scores=torch.cat([plausibility_scores,this_plausibility_scores],dim=0)

    plausible_article_index=torch.argmax(plausibility_scores,dim=0).item()
    return plausible_article_index

def get_most_plausible_chunk(
    model:Reader,
    tokenizer:AutoTokenizer,
    max_length:int,
    wikipedia_data_root_dir:Path,
    plausible_article_title:str,
    eval_batch_size:int,
    question:str):
    inputs=create_eval_model_inputs_2(
            tokenizer,
            max_length,
            wikipedia_data_root_dir,
            plausible_article_title,
            question
        )
    num_chunks:int=inputs["num_chunks"]
    num_sub_batches=num_chunks//eval_batch_size
    if num_chunks%eval_batch_size!=0:
        num_sub_batches+=1

    plausibility_scores=torch.empty(0)
    start_logits=torch.empty(0,max_length)
    end_logits=torch.empty(0,max_length)

    unk_id=tokenizer.convert_tokens_to_ids("[UNK]")

    for i in range(num_chunks):
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

            this_start_logits=this_start_logits.cpu()   #(1, eval_batch_size, sequence_length)
            this_end_logits=this_end_logits.cpu()   #(1, eval_batch_size, sequence_length)
            this_plausibility_scores=this_plausibility_scores.cpu() #(1, eval_batch_size)

            this_start_logits=torch.squeeze(this_start_logits)  #(eval_batch_size, sequence_length)
            this_end_logits=torch.squeeze(this_end_logits)  #(eval_batch_size, sequence_length)
            this_plausibility_scores=torch.squeeze(this_plausibility_scores)    #(eval_batch_size)

            this_start_logits=torch.softmax(this_start_logits,dim=1)
            this_end_logits=torch.softmax(this_end_logits,dim=1)

            start_logits=torch.cat([start_logits,this_start_logits],dim=0)
            end_logits=torch.cat([end_logits,this_end_logits],dim=0)

            plausibility_scores=torch.cat([plausibility_scores,this_plausibility_scores],dim=0)

    _,start_indices=torch.max(start_logits,dim=1)
    _,end_indices=torch.max(end_logits,dim=1)
    _,plausible_chunk_indices=torch.topk(plausibility_scores,k=num_chunks,dim=0)

    plausible_chunk_exists=False
    plausible_chunk_index=-1
    answer_start_index=-1
    answer_end_index=-1
    for i in range(num_chunks):
        plausible_chunk_index=plausible_chunk_indices[i].item()
        answer_start_index=start_indices[plausible_chunk_index].item()
        answer_end_index=end_indices[plausible_chunk_index].item()

        if answer_start_index==0 or answer_end_index==0:
                continue
        if answer_start_index>answer_end_index:
            continue
        if answer_end_index-answer_start_index>20:
            continue

        answer_ids=inputs["input_ids"][plausible_chunk_index,answer_start_index:answer_end_index+1].tolist()
        if unk_id in answer_ids:
            continue

        plausible_chunk_exists=True
        break

    predicted_answer="N/A"
    if plausible_chunk_exists:
        answer_ids=inputs["input_ids"][plausible_chunk_index,answer_start_index:answer_end_index+1].tolist()

        predicted_answer=tokenizer.decode(answer_ids)
        predicted_answer=predicted_answer.replace(" ","")

    ret={
        "predicted_answer":predicted_answer,
        "predicted_chunk_index":plausible_chunk_index
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

    qids:List[str]=[]
    questions:List[str]=[]
    answers:List[List[str]]=[]
    plausible_articles:List[str]=[]
    predicted_answers:List[str]=[]
    predicted_chunk_indices:List[int]=[]

    for batch in tqdm(eval_dataloader):
        qid=batch["qid"]
        question=batch["question"]
        this_answers=batch["answers"]
        top_k_titles=batch["top_k_titles"]
        top_k_scores=batch["top_k_scores"]

        if limit_num_top_k is not None:
            top_k_titles=top_k_titles[:limit_num_top_k]
            top_k_scores=top_k_scores[:limit_num_top_k]

        plausible_article_index=get_most_plausible_article_index(
            model,
            tokenizer,
            max_length,
            wikipedia_data_root_dir,
            eval_batch_size,
            question,
            top_k_titles,
            context_max_length
        )
        plausible_article_title=top_k_titles[plausible_article_index]
        
        plausible_chunk=get_most_plausible_chunk(
            model,
            tokenizer,
            max_length,
            wikipedia_data_root_dir,
            plausible_article_title,
            eval_batch_size,
            question
        )
        predicted_answer=plausible_chunk["predicted_answer"]
        predicted_chunk_index=plausible_chunk["predicted_chunk_index"]

        question_count+=1

        if predicted_answer in this_answers:
            correct_count+=1

        qids.append(qid)
        questions.append(question)
        answers.append(this_answers)
        plausible_articles.append(plausible_article_title)
        predicted_answers.append(predicted_answer)
        predicted_chunk_indices.append(predicted_chunk_index)

    ret={
        "accuracy":correct_count/question_count*100,

        "qids":qids,
        "questions":questions,
        "answers":answers,
        "plausible_articles":plausible_articles,
        "predicted_answers":predicted_answers,
        "predicted_chunk_indices":predicted_chunk_indices
    }
    return ret

def main(args):
    logger.info(args)

    eval_samples_filepath:str=args.eval_samples_filepath
    limit_num_eval_samples:int=args.limit_num_eval_samples
    reader_model_filepath:str=args.reader_model_filepath
    wikipedia_data_root_dirname:str=args.wikipedia_data_root_dirname
    bert_model_name:str=args.bert_model_name
    results_save_dirname:str=args.results_save_dirname
    eval_batch_size:int=args.eval_batch_size
    context_max_length:int=args.context_max_length
    limit_num_top_k:int=args.limit_num_top_k

    logger.info("モデルの評価を行う準備をしています...")

    results_save_dir=Path(results_save_dirname)
    results_save_dir.mkdir(parents=True,exist_ok=True)

    wikipedia_data_root_dir=Path(wikipedia_data_root_dirname)

    config=AutoConfig.from_pretrained(bert_model_name)
    tokenizer=AutoTokenizer.from_pretrained(bert_model_name)
    model=Reader(bert_model_name)

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
        plausible_articles=eval_results["plausible_articles"]
        predicted_answers=eval_results["predicted_answers"]
        predicted_chunk_indices=eval_results["predicted_chunk_indices"]

        for qid,question,this_answers,predicted_article,predicted_answer,predicted_chunk_index in zip(qids,questions,answers,plausible_articles,predicted_answers,predicted_chunk_indices):
            data={
                "qid":qid,
                "question":question,
                "answers":this_answers,
                "predicted_article":predicted_article,
                "predicted_answer":predicted_answer,
                "predicted_chunk_index":predicted_chunk_index
            }
            line=json.dumps(data,ensure_ascii=False)

            w.write(line)
            w.write("\n")

    logger.info("モデルの評価が完了しました")

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--eval_samples_filepath",type=str,default="../Data/Retriever/dev_top_ks.jsonl")
    parser.add_argument("--limit_num_eval_samples",type=int)
    parser.add_argument("--reader_model_filepath",type=str)
    parser.add_argument("--wikipedia_data_root_dirname",type=str,default="../Data/Wikipedia")
    parser.add_argument("--bert_model_name",type=str,default="cl-tohoku/bert-base-japanese-whole-word-masking")
    parser.add_argument("--results_save_dirname",type=str,default="../Data/Reader")
    parser.add_argument("--eval_batch_size",type=int,default=16)
    parser.add_argument("--context_max_length",type=int,default=3000)
    parser.add_argument("--limit_num_top_k",type=int)
    args=parser.parse_args()

    main(args)
