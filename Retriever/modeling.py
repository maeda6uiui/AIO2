import argparse
import hashlib
import json
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
from pathlib import Path
from transformers import AutoTokenizer,BertConfig,BertModel
from tqdm import tqdm
from typing import Dict,List

import sys
sys.path.append(".")

from models import RelevanceScoreCalculator

logging_fmt="%(asctime)s %(levelname)s: %(message)s"
logging.basicConfig(format=logging_fmt)
logger=logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class QAndDDataset(Dataset):
    def __init__(self):
        self.questions:List[str]=[]
        self.given_articles:List[str]=[]
        self.corresponding_flags:List[bool]=[]

    def __len__(self):
        return len(self.questions)

    def __getitem__(self,idx:int):
        question=self.questions[idx]
        given_article=self.given_articles[idx]
        corresponding_flag=self.corresponding_flags[idx]

        ret={
            "question":question,
            "given_article":given_article,
            "corresponding_flag":corresponding_flag
        }
        return ret

    def append(self,question:str,given_article:str,corresponding_flag:bool):
        self.questions.append(question)
        self.given_articles.append(given_article)
        self.corresponding_flags.append(corresponding_flag)

def create_dataset(samples_filepath:str)->QAndDDataset:
    dataset=QAndDDataset()

    with open(samples_filepath,"r") as r:
        for line in r:
            if line=="":
                continue

            data=json.loads(line)
            question=data["question"]
            given_article=data["given_article"]
            corresponding_flag=data["is_corresponding_article"]

            dataset.append(question,given_article,corresponding_flag)

    return dataset

def get_md5_hash(text:str)->str:
    return hashlib.md5(text.encode()).hexdigest()

def load_document_vectors(
    document_vectors_dir:Path,
    dim_document_vector:int,
    given_articles:List[str])->torch.FloatTensor:
    train_batch_size=len(given_articles)

    document_vectors=torch.empty(train_batch_size,dim_document_vector)

    for idx,given_article in enumerate(given_articles):
        article_hash=get_md5_hash(given_article)
        document_vector_file=document_vectors_dir.joinpath("{}.pt".format(article_hash))
        
        if not document_vector_file.exists():
            document_vectors[idx]=torch.zeros(dim_document_vector)
        else:
            document_vector=torch.load(document_vector_file,map_location=torch.device("cpu"))
            document_vectors[idx]=document_vector

    return document_vectors

def train(
    score_calculator:RelevanceScoreCalculator,
    bert:BertModel,
    train_dataloader:DataLoader,
    tokenizer:AutoTokenizer,
    document_vectors_dir:Path,
    max_length:int,
    dim_feature_vector:int,
    optimizer:optim.Optimizer,
    logging_steps:int)->float:
    score_calculator.train()

    criterion=nn.BCELoss()

    step_count=0
    total_loss=0

    for step,batch in enumerate(train_dataloader):
        questions=batch["question"]
        given_articles=batch["given_article"]
        corresponding_flags=batch["corresponding_flag"]

        #Questionの特徴量ベクトルを取得する
        q_inputs=tokenizer.encode_plus(
            questions,
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
            question_vectors=bert_outputs["pooler_output"] #(N, hidden_size)

        #Documentの特徴量ベクトルを取得する
        document_vectors=load_document_vectors(document_vectors_dir,dim_feature_vector,given_articles)
        document_vectors=document_vectors.to(device)

        #モデルの学習を行う
        score_calculator.zero_grad()

        logits=score_calculator(question_vectors,document_vectors)
        t_corresponding_flags=torch.as_tensor(corresponding_flags,dtype=torch.long,device=device)

        loss=criterion(logits,t_corresponding_flags)
        loss.backward()

        optimizer.step()

        step_count+=1
        total_loss+=loss.item()

        if step%logging_steps==0:
            logger.info("Step: {}\tLoss: {}".format(step,loss.item()))

        return total_loss/step_count

def eval(
    score_calculator:RelevanceScoreCalculator,
    bert:BertModel,
    eval_dataloader:DataLoader,
    tokenizer:AutoTokenizer,
    document_vectors_dir:Path,
    max_length:int,
    dim_feature_vector:int,
    results_save_file:Path)->Dict[float,float]:
    score_calculator.eval()

    criterion=nn.BCELoss()

    step_count=0
    total_loss=0

    all_questions:List[str]=[]
    all_given_articles:List[str]=[]
    all_corresponding_flags:List[bool]=[]
    all_logits=torch.empty(0,1)

    for batch in tqdm(eval_dataloader):
        questions=batch["question"]
        given_articles=batch["given_article"]
        corresponding_flags=batch["corresponding_flag"]

        #Questionの特徴量ベクトルを取得する
        q_inputs=tokenizer.encode_plus(
            questions,
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
            question_vectors=bert_outputs["pooler_output"] #(N, hidden_size)

        #Documentの特徴量ベクトルを取得する
        document_vectors=load_document_vectors(document_vectors_dir,dim_feature_vector,given_articles)
        document_vectors=document_vectors.to(device)

        #QuestionとDocumentの関連度を取得する
        with torch.no_grad():
            logits=score_calculator(question_vectors,document_vectors)
            
        t_corresponding_flags=torch.as_tensor(corresponding_flags,dtype=torch.long,device=device)
        loss=criterion(logits,t_corresponding_flags)

        step_count+=1
        total_loss+=loss.item()

        all_questions+=questions
        all_given_articles+=given_articles
        all_corresponding_flags+=corresponding_flags
        torch.cat([all_logits,logits.cpu()],dim=0)

    #正解率を計算する
    preds=(all_logits>0.5).long()   #(Number of questions, 1)
    preds=torch.squeeze(preds)  #(Number of questions)
    t_all_corresponding_flags=torch.as_tensor(all_corresponding_flags,dtype=torch.long)

    accuracy=(preds==t_all_corresponding_flags).mean().item()
    accuracy*=100

    logger.info("Accuracy: {}".format(accuracy))

    num_questions=len(all_questions)
    
    #結果をファイルに出力する
    with results_save_file.open("w") as w:
        w.write("Accuracy: {}\n".format(accuracy))
        w.write("\n")

        w.write("問題\t与えられた記事名\t関連している記事かどうか\tスコア\t予測結果\n")

        for i in range(num_questions):
            w.write("{}\t{}\t{}\t{}\t{}\n".format(
                all_questions[i],
                all_given_articles[i],
                all_corresponding_flags[i],
                all_logits[i].item(),
                all_logits[i].item()>0.5))

    ret={
        "accuracy":accuracy,
        "mean_loss":total_loss/step_count
    }
    return ret

def main(args):
    logger.info(args)

    train_samples_filepath:str=args.train_samples_filepath
    eval_samples_filepath:str=args.eval_samples_filepath
    document_vectors_dirname:str=args.document_vectors_dirname
    results_save_dirname:str=args.results_save_dirname
    bert_model_name:str=args.bert_model_name
    train_batch_size:int=args.train_batch_size
    eval_batch_size:int=args.eval_batch_size
    num_epochs:int=args.num_epochs
    resume_epoch:int=args.resume_epoch
    learning_rate:float=args.learning_rate
    logging_steps:int=args.logging_steps

    document_vectors_dir=Path(document_vectors_dirname)

    results_save_dir=Path(results_save_dirname)
    results_save_dir.mkdir(parents=True,exist_ok=True)

    logger.info("データローダを作成します...")

    train_dataset=create_dataset(train_samples_filepath)
    train_dataloader=DataLoader(train_dataset,batch_size=train_batch_size,shuffle=True)

    eval_dataset=create_dataset(eval_samples_filepath)
    eval_dataloader=DataLoader(eval_dataset,batch_size=eval_batch_size,shuffle=False)

    logger.info("モデル学習の準備を行っています...")

    config=BertConfig.from_pretrained(bert_model_name)
    score_calculator=RelevanceScoreCalculator(config.hidden_size)

    bert=BertModel.from_pretrained(bert_model_name)
    tokenizer=AutoTokenizer.from_pretrained(bert_model_name)

    if resume_epoch is not None:
        checkpoint_file=results_save_dir.joinpath("checkpoint_{}.pt".format(resume_epoch-1))
        logger.info("{}よりチェックポイントを読み込みます",str(checkpoint_file))

        state_dict=torch.load(checkpoint_file,map_location=torch.device("cpu"))
        score_calculator.load_state_dict(state_dict)

    score_calculator.to(device)

    bert.eval()
    bert.to(device)

    optimizer=optim.AdamW(score_calculator.parameters(),lr=learning_rate)

    start_epoch=resume_epoch if resume_epoch is not None else 0

    logger.info("モデルの学習を行っています...")

    for epoch in range(start_epoch,num_epochs):
        logger.info("===== {}/{} =====".format(epoch,num_epochs-1))

        mean_loss=train(
            score_calculator,
            bert,
            train_dataloader,
            tokenizer,
            document_vectors_dir,
            config.max_length,
            config.hidden_size,
            optimizer,
            logging_steps)

        logger.info("学習時の平均損失: {}".format(mean_loss))

        checkpoint_save_file=results_save_dir.joinpath("checkpoint_{}.pt".format(epoch))
        torch.save(score_calculator.state_dict(),checkpoint_save_file)

        eval_results_save_file=results_save_dir.joinpath("results_{}.txt".format(epoch))

        eval_results=eval(
            score_calculator,
            bert,
            eval_dataloader,
            tokenizer,
            document_vectors_dir,
            config.max_length,
            config.hidden_size,
            eval_results_save_file)
        eval_accuracy=eval_results["accuracy"]
        eval_mean_loss=eval_results["mean_loss"]

        logger.info("評価時の平均損失: {}\t正解率: {}".format(eval_mean_loss,eval_accuracy))

    logger.info("モデルの学習が完了しました")

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--train_samples_filepath",type=str,default="../Data/train_samples.jsonl")
    parser.add_argument("--document_vectors_dirname",type=str,default="../Data/WikipediaVector")
    parser.add_argument("--results_save_dirname",type=str,default="../Data/Retriever")
    parser.add_argument("--bert_model_name",type=str,default="cl-tohoku/bert-base-japanese-whole-word-masking")
    parser.add_argument("--train_batch_size",type=int,default=128)
    parser.add_argument("--eval_batch_size",type=int,default=128)
    parser.add_argument("--num_epochs",type=int,default=5)
    parser.add_argument("--resume_epoch",type=int)
    parser.add_argument("--learning_rate",type=float,default=1e-4)
    parser.add_argument("--logging_steps",type=int,default=10)
    args=parser.parse_args()

    main(args)
