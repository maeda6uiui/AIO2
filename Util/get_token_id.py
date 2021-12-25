import argparse
from transformers import AutoTokenizer

def main(args):
    bert_model_name:str=args.bert_model_name
    token:str=args.token

    tokenizer=AutoTokenizer.from_pretrained(bert_model_name)

    token_id=tokenizer.convert_tokens_to_ids(token)

    print("{}={}".format(token,token_id))

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--bert_model_name",type=str,default="cl-tohoku/bert-base-japanese-whole-word-masking")
    parser.add_argument("-t","--token",type=str)
    args=parser.parse_args()

    main(args)
