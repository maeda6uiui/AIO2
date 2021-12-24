import argparse
from transformers import AutoConfig,AutoTokenizer,AutoModel

def main(args):
    bert_model_name:str=args.bert_model_name

    config=AutoConfig.from_pretrained(bert_model_name)
    tokenizer=AutoTokenizer.from_pretrained(bert_model_name)
    model=AutoModel.from_pretrained(bert_model_name)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--bert_model_name",type=str,default="cl-tohoku/bert-base-japanese-whole-word-masking")
    args=parser.parse_args()

    main(args)
