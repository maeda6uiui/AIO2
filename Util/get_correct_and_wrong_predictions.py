import argparse
import json
from typing import List,Tuple

class Predictions(object):
    def __init__(self):
        self.qids:List[str]=[]
        self.questions:List[str]=[]
        self.answers:List[List[str]]=[]
        self.predicted_articles:List[str]=[]
        self.predicted_answers:List[str]=[]

    def append(self,qid:str,question:str,this_answers:List[str],predicted_article:str,predicted_answer:str):
        self.qids.append(qid)
        self.questions.append(question)
        self.answers.append(this_answers)
        self.predicted_articles.append(predicted_article)
        self.predicted_answers.append(predicted_answer)

    def get_items(self)->Tuple[List[str],List[str],List[List[str]],List[str],List[str]]:
        return zip(self.qids,self.questions,self.answers,self.predicted_articles,self.predicted_answers)

def load_predictions(input_filepath:str)->Predictions:
    predictions=Predictions()

    with open(input_filepath,"r") as r:
        for line in r:
            data=json.loads(line)

            qid=data["qid"]
            question=data["question"]
            this_answers=data["answers"]
            predicted_article=data["predicted_article"]
            predicted_answer=data["predicted_answer"]

            predictions.append(qid,question,this_answers,predicted_article,predicted_answer)

    return predictions

def save_predictions(output_filepath:str,predictions:Predictions):
    with open(output_filepath,"w") as w:
        for qid,question,this_answers,predicted_article,predicted_answer in predictions.get_items():
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

def main(args):
    input_filepath:str=args.input_filepath
    correct_output_filepath:str=args.correct_output_filepath
    wrong_output_filepath:str=args.wrong_output_filepath

    predictions=load_predictions(input_filepath)

    correct_predictions=Predictions()
    wrong_predictions=Predictions()

    for qid,question,this_answers,predicted_article,predicted_answer in predictions.get_items():
        if predicted_answer in this_answers:
            correct_predictions.append(qid,question,this_answers,predicted_article,predicted_answer)
        else:
            wrong_predictions.append(qid,question,this_answers,predicted_article,predicted_answer)

    save_predictions(correct_output_filepath,correct_predictions)
    save_predictions(wrong_output_filepath,wrong_predictions)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--input_filepath",type=str)
    parser.add_argument("--correct_output_filepath",type=str)
    parser.add_argument("--wrong_output_filepath",type=str)
    args=parser.parse_args()

    main(args)
