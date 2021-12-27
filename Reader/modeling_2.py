import argparse
import hashlib
import json
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
from pathlib import Path
from transformers import AutoConfig,AutoTokenizer,BertForQuestionAnswering
from tqdm import tqdm
from typing import List

import sys
sys.path.append(".")
from models import PlausibilityCalculator

logging_fmt="%(asctime)s %(levelname)s: %(message)s"
logging.basicConfig(format=logging_fmt)
logger=logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReaderTrainDataset(Dataset):
    def __init__(self):
        self.questions:List[str]=[]
        self.article_titles:List[str]=[]
        self.start_indices:List[int]=[]
        self.end_indices:List[int]=[]

    def __len__(self):
        return len(self.questions)

    def __getitem__(self,idx:int):
        question=self.questions[idx]
        article_title=self.article_titles[idx]
        start_index=self.start_indices[idx]
        end_index=self.end_indices[idx]

        ret={
            "question":question,
            "article_title":article_title,
            "start_index":start_index,
            "end_index":end_index
        }
        return ret

    def append(self,question:str,article_title:str,answer_ranges:List[str]):
        for answer_range in answer_ranges:
            self.questions.append(question)
            self.article_titles.append(article_title)

            start_index,end_index=answer_range.split("-")
            start_index=int(start_index)
            end_index=int(end_index)

            self.start_indices.append(start_index)
            self.end_indices.append(end_index)

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

def create_train_dataset(samples_filepath:str,limit_num_samples:int=None)->ReaderTrainDataset:
    dataset=ReaderTrainDataset()

    with open(samples_filepath,"r") as r:
        lines=r.read().splitlines()

    if limit_num_samples is not None:
        lines=lines[:limit_num_samples]

    for line in lines:
        data=json.loads(line)

        question=data["question"]
        article_title=data["article_title"]
        this_answer_ranges=data["answer_ranges"]

        dataset.append(question,article_title,this_answer_ranges)

    return dataset

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

def create_train_model_inputs(
    tokenizer:AutoTokenizer,
    max_length:int,
    wikipedia_data_root_dir:Path,
    questions:List[str],
    article_titles:List[str],
    start_indices:List[int],
    end_indices:List[int],
    context_max_length:int):
    contexts:List[str]=[]
    for article_title in article_titles:
        title_hash=get_md5_hash(article_title)
        text_file=wikipedia_data_root_dir.joinpath(title_hash,"text.txt")

        with text_file.open("r") as r:
            context=r.read()
            context=context[:context_max_length]
            contexts.append(context)

    batch_size=len(questions)

    input_ids=torch.empty(batch_size,max_length,dtype=torch.long)
    attention_mask=torch.empty(batch_size,max_length,dtype=torch.long)
    token_type_ids=torch.empty(batch_size,max_length,dtype=torch.long)

    for i in range(batch_size):
        encode=tokenizer.encode_plus(
            questions[i],
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

    start_positions=torch.tensor(start_indices,dtype=torch.long,device=device)
    end_positions=torch.tensor(end_indices,dtype=torch.long,device=device)

    ret={
        "input_ids":input_ids,
        "attention_mask":attention_mask,
        "token_type_ids":token_type_ids,
        "start_positions":start_positions,
        "end_positions":end_positions
    }
    return ret

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

def train(
    model:BertForQuestionAnswering,
    pcalculator:PlausibilityCalculator,
    train_dataloader:DataLoader,
    optimizer:optim.Optimizer,
    tokenizer:AutoTokenizer,
    max_length:int,
    wikipedia_data_root_dir:Path,
    logging_steps:int,
    context_max_length:int):
    model.train()
    pcalculator.train()

    criterion_pcalculator=nn.BCELoss()

    step_count=0
    total_loss_model=0
    total_loss_pcalculator=0

    for step,batch in enumerate(train_dataloader):
        questions=batch["question"]
        article_titles=batch["article_title"]
        start_indices=batch["start_index"]
        end_indices=batch["end_index"]

        inputs=create_train_model_inputs(
            tokenizer,
            max_length,
            wikipedia_data_root_dir,
            questions,
            article_titles,
            start_indices,
            end_indices,
            context_max_length
        )
        inputs["return_dict"]=True
        inputs["output_hidden_states"]=True

        #===== Model =====
        model.zero_grad()

        outputs=model(**inputs)
        
        loss_model=outputs["loss"]
        total_loss_model+=loss_model.item()

        #==========
        #===== Plausibility Calculator =====
        pcalculator.zero_grad()

        hidden_states=outputs["hidden_states"]
        cls_output=hidden_states[12][:,0,:]  #(N, hidden_size)

        plausibility_scores=pcalculator(cls_output) #(N, 1)
        plausibility_scores=torch.squeeze(plausibility_scores)  #(N)

        start_positions=inputs["start_positions"]
        targets=(start_positions!=0).float()
        loss_pcalculator=criterion_pcalculator(plausibility_scores,targets)
        total_loss_pcalculator+=loss_pcalculator.item()

        #==========

        loss=loss_model+loss_pcalculator
        loss.backward()

        optimizer.step()

        step_count+=1

        if step%logging_steps==0:
            logger.info("Step: {}\tLoss (Total): {}\tLoss (Model): {}\tLoss (PCalculator): {}".format(
                    step,
                    loss.item(),
                    loss_model.item(),
                    loss_pcalculator.item()))

    ret={
        "mean_loss":(total_loss_model+total_loss_pcalculator)/step_count,
        "mean_loss_model":total_loss_model/step_count,
        "mean_loss_pcalculator":total_loss_pcalculator/step_count
    }
    return ret

def eval(
    model:BertForQuestionAnswering,
    pcalculator:PlausibilityCalculator,
    eval_dataloader:DataLoader,
    tokenizer:AutoTokenizer,
    max_length:int,
    wikipedia_data_root_dir:Path,
    eval_batch_size:int,
    context_max_length:int,
    limit_num_top_k:int,
    mul_retrieval_scores:bool):
    model.eval()
    pcalculator.eval()

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
                "token_type_ids":token_type_ids,
                "return_dict":True,
                "output_hidden_states":True
            }

            with torch.no_grad():
                outputs=model(**sub_inputs)

                this_start_logits=outputs["start_logits"]
                this_end_logits=outputs["end_logits"]

                this_start_logits=this_start_logits.cpu()
                this_end_logits=this_end_logits.cpu()

                this_start_logits=torch.softmax(this_start_logits,dim=1)
                this_end_logits=torch.softmax(this_end_logits,dim=1)

                start_logits=torch.cat([start_logits,this_start_logits],dim=0)
                end_logits=torch.cat([end_logits,this_end_logits],dim=0)

                hidden_states=outputs["hidden_states"]
                cls_output=hidden_states[12][:,0,:]  #(N, hidden_size)

                this_plausibility_scores=pcalculator(cls_output) #(N, 1)
                this_plausibility_scores=this_plausibility_scores.cpu()
                this_plausibility_scores=torch.squeeze(this_plausibility_scores)  #(N)

                plausibility_scores=torch.cat([plausibility_scores,this_plausibility_scores],dim=0)

        _,start_indices=torch.max(start_logits,dim=1)
        _,end_indices=torch.max(end_logits,dim=1)

        if mul_retrieval_scores:
            for i in range(num_titles):
                plausibility_scores[i]*=top_k_scores[i]

        _,plausible_article_indices=torch.topk(plausibility_scores,k=num_titles,dim=0)

        plausible_article_exists=False
        plausible_article_index=-1
        answer_start_index=-1
        answer_end_index=-1
        for i in range(num_titles):
            plausible_article_index=plausible_article_indices[i].item()
            answer_start_index=start_indices[plausible_article_index].item()
            answer_end_index=end_indices[plausible_article_index].item()

            if answer_start_index!=0 and answer_start_index<=answer_end_index:
                plausible_article_exists=True
                break

        predicted_answer="N/A"
        predicted_article="N/A"
        if plausible_article_exists:
            predicted_article=top_k_titles[plausible_article_index]

            answer_ids=inputs["input_ids"][plausible_article_index,answer_start_index:answer_end_index+1].tolist()

            #予測結果に[UNK]が含まれる場合は、再度Tokenizeを行って対応するトークンを取得する
            if unk_id in answer_ids:
                question_tokens=tokenizer.tokenize(question)

                title_hash=get_md5_hash(predicted_article)
                text_file=wikipedia_data_root_dir.joinpath(title_hash,"text.txt")
                with text_file.open("r") as r:
                    context=r.read()[:context_max_length]

                context_tokens=tokenizer.tokenize(context)

                tokens=["[CLS]"]+question_tokens+["[SEP]"]+context_tokens+["[SEP]"]

                predicted_answer="".join(tokens[answer_start_index:answer_end_index+1])
                predicted_answer=predicted_answer.replace("#","")
            else:
                predicted_answer=tokenizer.decode(answer_ids)
                predicted_answer=predicted_answer.replace(" ","")

            if predicted_answer in this_answers:
                correct_count+=1

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

    train_samples_filepath:str=args.train_samples_filepath
    eval_samples_filepath:str=args.eval_samples_filepath
    limit_num_train_samples:int=args.limit_num_train_samples
    limit_num_eval_samples:int=args.limit_num_eval_samples
    wikipedia_data_root_dirname:str=args.wikipedia_data_root_dirname
    bert_model_name:str=args.bert_model_name
    results_save_dirname:str=args.results_save_dirname
    learning_rate:float=args.learning_rate
    num_epochs:int=args.num_epochs
    resume_epoch:int=args.resume_epoch
    train_batch_size:int=args.train_batch_size
    eval_batch_size:int=args.eval_batch_size
    logging_steps:int=args.logging_steps
    context_max_length:int=args.context_max_length
    limit_num_top_k:int=args.limit_num_top_k
    mul_retrieval_scores:bool=args.mul_retrieval_scores

    logger.info("モデルの学習を行う準備をしています...")

    results_save_dir=Path(results_save_dirname)
    results_save_dir.mkdir(parents=True,exist_ok=True)

    wikipedia_data_root_dir=Path(wikipedia_data_root_dirname)

    config=AutoConfig.from_pretrained(bert_model_name)
    tokenizer=AutoTokenizer.from_pretrained(bert_model_name)
    model=BertForQuestionAnswering.from_pretrained(bert_model_name)

    pcalculator=PlausibilityCalculator(config.hidden_size)

    if resume_epoch is not None:
        checkpoint_file=results_save_dir.joinpath("checkpoint_{}.pt".format(resume_epoch-1))
        state_dict=torch.load(checkpoint_file,map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)

        logger.info("{}よりチェックポイント(BertForQuestionAnswering)を読み込みました".format(str(checkpoint_file)))

        pcalculator_file=results_save_dir.joinpath("pcalculator_{}.pt".format(resume_epoch-1))
        state_dict=torch.load(pcalculator_file,map_location=torch.device("cpu"))
        pcalculator.load_state_dict(state_dict)

        logger.info("{}よりチェックポイント(Plausibility Calculator)を読み込みました".format(str(pcalculator_file)))

    model.to(device)
    pcalculator.to(device)

    parameters=list(model.parameters())+list(pcalculator.parameters())
    optimizer=optim.AdamW(parameters,lr=learning_rate)

    train_dataset=create_train_dataset(train_samples_filepath,limit_num_samples=limit_num_train_samples)
    train_dataloader=DataLoader(train_dataset,batch_size=train_batch_size,shuffle=True)

    logger.info("学習データの数: {}".format(len(train_dataset)))

    eval_dataset=create_eval_dataset(eval_samples_filepath,limit_num_samples=limit_num_eval_samples)
    eval_dataloader=DataLoader(eval_dataset,batch_size=1,shuffle=False,collate_fn=collate_fn_for_eval_dataset)

    logger.info("評価データの数: {}".format(len(eval_dataset)))

    logger.info("モデルの学習を開始します")

    start_epoch=resume_epoch if resume_epoch is not None else 0

    for epoch in range(start_epoch,num_epochs):
        logger.info("===== {}/{} =====".format(epoch,num_epochs-1))

        train_results=train(
            model,
            pcalculator,
            train_dataloader,
            optimizer,
            tokenizer,
            config.max_position_embeddings,
            wikipedia_data_root_dir,
            logging_steps,
            context_max_length
        )
        logger.info("エポック{}の学習が終了しました".format(epoch))

        checkpoint_file=results_save_dir.joinpath("checkpoint_{}.pt".format(epoch))
        torch.save(model.state_dict(),checkpoint_file)

        pcalculator_file=results_save_dir.joinpath("pcalculator_{}.pt".format(epoch))
        torch.save(pcalculator.state_dict(),pcalculator_file)

        train_mean_loss=train_results["mean_loss"]
        train_mean_loss_model=train_results["mean_loss_model"]
        train_mean_loss_pcalculator=train_results["mean_loss_pcalculator"]

        logger.info("平均損失(合計): {}\t平均損失(モデル): {}\t平均損失(PCalculator): {}".format(
            train_mean_loss,train_mean_loss_model,train_mean_loss_pcalculator))

        eval_results=eval(
            model,
            pcalculator,
            eval_dataloader,
            tokenizer,
            config.max_position_embeddings,
            wikipedia_data_root_dir,
            eval_batch_size,
            context_max_length,
            limit_num_top_k,
            mul_retrieval_scores
        )
        logger.info("エポック{}の評価が終了しました".format(epoch))

        eval_accuracy=eval_results["accuracy"]

        logger.info("正解率(完全一致): {} %".format(eval_accuracy))

        eval_result_file=results_save_dir.joinpath("eval_result_{}.txt".format(epoch))
        with eval_result_file.open("w") as w:
            w.write("正解率(完全一致): {} %\n".format(eval_accuracy))

        eval_predictions_file=results_save_dir.joinpath("eval_predictions_{}.jsonl".format(epoch))
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

    logger.info("モデルの学習が完了しました")

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--train_samples_filepath",type=str,default="../Data/Reader/train_samples.jsonl")
    parser.add_argument("--eval_samples_filepath",type=str,default="../Data/Retriever/dev_top_ks.jsonl")
    parser.add_argument("--limit_num_train_samples",type=int)
    parser.add_argument("--limit_num_eval_samples",type=int)
    parser.add_argument("--wikipedia_data_root_dirname",type=str,default="../Data/Wikipedia")
    parser.add_argument("--bert_model_name",type=str,default="cl-tohoku/bert-base-japanese-whole-word-masking")
    parser.add_argument("--results_save_dirname",type=str,default="../Data/Reader")
    parser.add_argument("--learning_rate",type=float,default=1e-5)
    parser.add_argument("--num_epochs",type=int,default=5)
    parser.add_argument("--resume_epoch",type=int)
    parser.add_argument("--train_batch_size",type=int,default=16)
    parser.add_argument("--eval_batch_size",type=int,default=16)
    parser.add_argument("--logging_steps",type=int,default=100)
    parser.add_argument("--context_max_length",type=int,default=3000)
    parser.add_argument("--limit_num_top_k",type=int)
    parser.add_argument("--mul_retrieval_scores",action="store_true")
    args=parser.parse_args()

    main(args)
