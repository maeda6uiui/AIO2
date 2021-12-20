import argparse
import hashlib
import json
import logging
import torch
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
from pathlib import Path
from transformers import AutoConfig,AutoTokenizer,BertForQuestionAnswering
from typing import List

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

        ret={
            "qid":qid,
            "question":question,
            "answers":this_answers,
            "top_k_titles":this_top_k_titles,
            "top_k_scores":this_top_k_scores
        }
        return ret

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

def create_eval_dataset(samples_filepath:str)->ReaderEvalDataset:
    dataset=ReaderEvalDataset()

    with open(samples_filepath,"r") as r:
        for line in r:
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
    end_indices:List[int]):
    contexts:List[str]=[]
    for article_title in article_titles:
        title_hash=get_md5_hash(article_title)
        text_file=wikipedia_data_root_dir.joinpath(title_hash,"text.txt")

        with text_file.open("r") as r:
            context=r.read()
            contexts.append(context)

    encode=tokenizer.encode_plus(
        questions,
        contexts,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt"
    )

    input_ids=encode["input_ids"].to(device)
    attention_mask=encode["attention_mask"].to(device)
    token_type_ids=encode["token_type_ids"].to(device)

    start_positions=torch.LongTensor(start_indices,device=device) if start_indices is not None else None
    end_positions=torch.LongTensor(end_indices,device=device) if end_indices is not None else None

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
    top_k_titles:List[str]):
    contexts:List[str]=[]
    for article_title in top_k_titles:
        title_hash=get_md5_hash(article_title)
        text_file=wikipedia_data_root_dir.joinpath(title_hash,"text.txt")

        with text_file.open("r") as r:
            context=r.read()
            contexts.append(context)

    questions=[]
    for i in range(len(top_k_titles)):
        questions.append(question)

    encode=tokenizer.encode_plus(
        questions,
        contexts,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt"
    )

    input_ids=encode["input_ids"].to(device)
    attention_mask=encode["attention_mask"].to(device)
    token_type_ids=encode["token_type_ids"].to(device)

    ret={
        "input_ids":input_ids,
        "attention_mask":attention_mask,
        "token_type_ids":token_type_ids
    }
    return ret

def train(
    model:BertForQuestionAnswering,
    train_dataloader:DataLoader,
    optimizer:optim.Optimizer,
    tokenizer:AutoTokenizer,
    max_length:int,
    wikipedia_data_root_dir:Path,
    logging_steps:int)->float:
    model.train()

    step_count=0
    total_loss=0

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
            end_indices
        )
        inputs["return_dict"]=True

        model.zero_grad()

        outputs=model(**inputs)
        
        loss=outputs["loss"]
        loss.backward()

        optimizer.step()

        step_count+=1
        total_loss+=loss.item()

        if step%logging_steps==0:
            logger.info("Step: {}\tLoss: {}".format(step,loss.item()))

    return total_loss/step_count

def eval(
    model:BertForQuestionAnswering,
    eval_dataloader:DataLoader,
    tokenizer:AutoTokenizer,
    max_length:int,
    wikipedia_data_root_dir:Path,
    eval_batch_size:int):
    model.eval()

    question_count=0
    correct_count=0

    qids:List[str]=[]
    questions:List[str]=[]
    answers:List[List[str]]=[]
    predictions:List[str]=[]

    #問題1問を一つのバッチとするため、評価用データローダのバッチサイズは1で固定しておく
    #コマンドライン引数で設定するバッチサイズ(eval_batch_size)は、
    #一度にいくつの記事をモデルに入力するかを指定するためのもの
    for batch in eval_dataloader:
        qid=batch["qid"][0]
        question=batch["question"][0]
        this_answers=batch["answers"][0]
        top_k_titles=batch["top_k_titles"][0]
        top_k_scores=batch["top_k_scores"][0]

        inputs=create_eval_model_inputs(
            tokenizer,
            max_length,
            wikipedia_data_root_dir,
            question,
            top_k_titles
        )

        num_sub_batches=len(top_k_titles)//eval_batch_size
        if len(top_k_titles)%eval_batch_size!=0:
            num_sub_batches+=1

        plausibility_scores=torch.empty(0)
        start_indices=torch.empty(0,dtype=torch.long)
        end_indices=torch.empty(0,dtype=torch.long)

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
                "return_dict":True
            }

            with torch.no_grad():
                outputs=model(**sub_inputs)

                start_logits=outputs["start_logits"]
                end_logits=outputs["end_logits"]

                start_logits=start_logits.cpu()
                end_logits=end_logits.cpu()

                this_start_scores,this_start_indices=torch.max(start_logits,dim=0)
                this_end_scores,this_end_indices=torch.max(end_logits,dim=0)

                start_indices=torch.cat([start_indices,this_start_indices],dim=0)
                end_indices=torch.cat([end_indices,this_end_indices],dim=0)
                
                this_plausibility_scores=torch.mul(this_start_scores,this_end_scores)
                plausibility_scores=torch.cat([plausibility_scores,this_plausibility_scores],dim=0)

        for i in range(len(top_k_titles)):
            plausibility_scores[i]*=top_k_scores[i]

        most_plausible_answer_index=torch.argmax(plausibility_scores,dim=0).item()
        answer_start_index=start_indices[most_plausible_answer_index].item()
        answer_end_index=end_indices[most_plausible_answer_index].item()

        answer_ids=inputs["input_ids"][most_plausible_answer_index,answer_start_index:answer_end_index+1]

        predicted_answer=tokenizer.decode(answer_ids)
        predicted_answer=predicted_answer.replace(" ","")

        if predicted_answer in this_answers:
            correct_count+=1

        question_count+=1

        qids.append(qid)
        questions.append(question)
        answers.append(this_answers)
        predictions.append(predicted_answer)

    ret={
        "accuracy":correct_count/question_count*100,
        "qids":qids,
        "questions":questions,
        "answers":answers,
        "predictions":predictions
    }
    return ret

def main(args):
    logger.info(args)

    train_samples_filepath:str=args.train_samples_filepath
    eval_samples_filepath:str=args.eval_samples_filepath
    limit_num_train_samples:int=args.limit_num_train_samples
    wikipedia_data_root_dirname:str=args.wikipedia_data_root_dirname
    bert_model_name:str=args.bert_model_name
    results_save_dirname:str=args.results_save_dirname
    learning_rate:float=args.learning_rate
    num_epochs:int=args.num_epochs
    resume_epoch:int=args.resume_epoch
    train_batch_size:int=args.train_batch_size
    eval_batch_size:int=args.eval_batch_size
    logging_steps:int=args.logging_steps

    logger.info("モデルの学習を行う準備をしています...")

    results_save_dir=Path(results_save_dirname)
    results_save_dir.mkdir(parents=True,exist_ok=True)

    wikipedia_data_root_dir=Path(wikipedia_data_root_dirname)

    config=AutoConfig.from_pretrained(bert_model_name)
    tokenizer=AutoTokenizer.from_pretrained(bert_model_name)
    model=BertForQuestionAnswering.from_pretrained(bert_model_name)

    if resume_epoch is not None:
        checkpoint_file=results_save_dir.joinpath("checkpoint_{}.pt".format(resume_epoch-1))
        state_dict=torch.load(checkpoint_file,map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)

        logger.info("{}よりチェックポイントを読み込みました".format(str(checkpoint_file)))

    model.to(device)

    optimizer=optim.AdamW(model.parameters(),lr=learning_rate)

    train_dataset=create_train_dataset(train_samples_filepath,limit_num_samples=limit_num_train_samples)
    train_dataloader=DataLoader(train_dataset,batch_size=train_batch_size,shuffle=True)

    logger.info("学習データの数: {}".format(len(train_dataset)))

    eval_dataset=create_eval_dataset(eval_samples_filepath)
    eval_dataloader=DataLoader(eval_dataset,batch_size=1,shuffle=False)

    logger.info("評価データの数: {}".format(len(eval_dataset)))

    logger.info("モデルの学習を開始します")

    start_epoch=resume_epoch if resume_epoch is not None else 0

    for epoch in range(start_epoch,num_epochs):
        logger.info("===== {}/{} =====".format(epoch,num_epochs-1))

        train_mean_loss=train(
            model,
            train_dataloader,
            optimizer,
            tokenizer,
            config.max_length,
            wikipedia_data_root_dir,
            logging_steps
        )
        logger.info("学習時の平均損失: {}".format(train_mean_loss))

        checkpoint_file=results_save_dir.joinpath("checkpoint_{}.pt".format(epoch))
        torch.save(model.state_dict(),checkpoint_file)

        eval_results=eval(
            model,
            eval_dataloader,
            tokenizer,
            config.max_length,
            wikipedia_data_root_dir,
            eval_batch_size
        )

        eval_accuracy=eval_results["accuracy"]

        logger.info("評価時の正解率(完全一致): {} %".format(eval_accuracy))

        eval_result_file=results_save_dir.joinpath("eval_result_{}.txt".format(epoch))
        with eval_result_file.open("w") as w:
            w.write("評価時の正解率(完全一致): {} %\n".format(eval_accuracy))

        eval_predictions_file=results_save_dir.joinpath("eval_predictions_{}.jsonl".format(epoch))
        with eval_predictions_file.open("w") as w:
            qids=eval_results["qids"]
            questions=eval_results["questions"]
            answers=eval_results["answers"]
            predictions=eval_results["predictions"]

            for qid,question,this_answers,prediction in zip(qids,questions,answers,predictions):
                data={
                    "qid":qid,
                    "question":question,
                    "answers":this_answers,
                    "prediction":prediction
                }
                line=json.dumps(data,ensure_ascii=False)

                w.write(line)
                w.write("\n")

    logger.info("モデルの学習が完了しました")

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--train_samples_filepath",type=str,default="../Data/Reader/train_samples.jsonl")
    parser.add_argument("--eval_samples_filepath",type=str,default="../Data/Retriever/dev_top_k_samples.jsonl")
    parser.add_argument("--limit_num_train_samples",type=int)
    parser.add_argument("--wikipedia_data_root_dirname",type=str,default="../Data/Wikipedia")
    parser.add_argument("--bert_model_name",type=str,default="cl-tohoku/bert-base-japanese-whole-word-masking")
    parser.add_argument("--results_save_dirname",type=str,default="../Data/Reader")
    parser.add_argument("--learning_rate",type=float,default=1e-5)
    parser.add_argument("--num_epochs",type=int,default=5)
    parser.add_argument("--resume_epoch",type=int)
    parser.add_argument("--train_batch_size",type=int,default=16)
    parser.add_argument("--eval_batch_size",type=int,default=16)
    parser.add_argument("--logging_steps",type=int,default=100)
    args=parser.parse_args()

    main(args)
