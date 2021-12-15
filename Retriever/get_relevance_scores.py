import argparse
import logging
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import BertConfig,BertModel,AutoTokenizer

import sys
sys.path.append(".")

from models import RelevanceScoreCalculator

logging_fmt="%(asctime)s %(levelname)s: %(message)s"
logging.basicConfig(format=logging_fmt)
logger=logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args):
    logger.info(args)

    input_text:str=args.input_text
    wikipedia_data_root_dirname:str=args.wikipedia_data_root_dirname
    output_filepath:str=args.output_filepath
    bert_model_name:str=args.bert_model_name
    score_calculator_filepath:str=args.score_calculator_filepath

    logger.info("関連度スコア計算の準備を行っています...")

    wikipedia_data_root_dir=Path(wikipedia_data_root_dirname)
    wikipedia_data_dirs=wikipedia_data_root_dir.glob("*")
    wikipedia_data_dirs=list(wikipedia_data_dirs)

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

    encode=tokenizer.encode_plus(
        input_text,
        padding="max_length",
        max_length=config.max_length,
        truncation=True,
        return_tensors="pt")
    bert_inputs={
        "input_ids":encode["input_ids"].to(device),
        "attention_mask":encode["attention_mask"].to(device),
        "token_type_ids":encode["token_type_ids"].to(device),
        "return_dict":True
    }
    with torch.no_grad():
        bert_outputs=bert(**bert_inputs)
        input_text_vector=bert_outputs["pooler_output"]

    num_wikipedia_articles=len(wikipedia_data_dirs)
    scores=torch.empty(num_wikipedia_articles,device=device)

    logger.info("関連度スコアの計算を行っています...")

    for idx,wikipedia_data_dir in enumerate(tqdm(wikipedia_data_dirs)):
        document_vector_file=wikipedia_data_dir.joinpath("vector.pt")
        document_vector=torch.load(document_vector_file,map_location=torch.device("cpu"))
        document_vector=document_vector.to(device)

        with torch.no_grad():
            score=score_calculator(input_text_vector,document_vector)
            score=torch.squeeze(score)
            scores[idx]=score

    sorted_scores,sorted_indices=torch.topk(scores,k=num_wikipedia_articles)

    logger.info("結果をファイルに出力しています...")

    with open(output_filepath,"w") as w:
        w.write("入力テキスト: {}\n".format(input_text))
        w.write("\n")

        for i in range(num_wikipedia_articles):
            wikipedia_data_dir:Path=wikipedia_data_dirs[sorted_indices[i].item()]
            title_file=wikipedia_data_dir.joinpath("title.txt")
            with title_file.open("r") as r:
                title=r.read().splitlines()[0]

            w.write("{}\t{}\n".format(title,sorted_scores[i].item()))

    logger.info("処理が完了しました")

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--input_text",type=str)
    parser.add_argument("--wikipedia_data_root_dirname",type=str,default="../Data/Wikipedia")
    parser.add_argument("--output_filepath",type=str,default="../Data/relevance_scores.txt")
    parser.add_argument("--bert_model_name",type=str,default="cl-tohoku/bert-base-japanese-whole-word-masking")
    parser.add_argument("--score_calculator_filepath",type=str,default="../Data/Retriever/checkpoint_7.pt")
    args=parser.parse_args()

    main(args)
