import argparse
import json

def main(args):
    input_filepath:str=args.input_filepath

    min_num_negative_articles=1000
    with open(input_filepath,"r") as r:
        for line in r:
            data=json.loads(line)

            negative_article_titles=data["negative_article_titles"]
            if len(negative_article_titles)<min_num_negative_articles:
                min_num_negative_articles=len(negative_article_titles)

    print(min_num_negative_articles)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("-i","--input_filepath",type=str,default="../Data/Reader/train_samples.jsonl")
    args=parser.parse_args()

    main(args)
