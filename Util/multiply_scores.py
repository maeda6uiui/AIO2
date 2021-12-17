import argparse

def main(args):
    #関連度スコアが列挙されたファイルについて、
    #あらかじめ記事名でソートしておくこと
    input_filepath_1:str=args.input_filepath_1
    input_filepath_2:str=args.input_filepath_2
    output_filepath:str=args.output_filepath

    with open(input_filepath_1,"r") as r:
        lines_1=r.read().splitlines()

    with open(input_filepath_2,"r") as r:
        lines_2=r.read().splitlines()

    with open(output_filepath,"w") as w:
        for line_1,line_2 in zip(lines_1,lines_2):
            splits_1=line_1.split("\t")
            splits_2=line_2.split("\t")

            title=splits_1[0]

            score_1=float(splits_1[1])
            score_2=float(splits_2[1])

            score=score_1*score_2

            w.write("{}\t{}\n".format(title,score))

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--input_filepath_1",type=str,default="../Data/Retriever/relevance_scores_bert.txt")
    parser.add_argument("--input_filepath_2",type=str,default="../Data/Retriever/relevance_scores_tfidf.txt")
    parser.add_argument("--output_filepath",type=str,default="../Data/Retriever/relevance_scores_mul.txt")
    args=parser.parse_args()

    main(args)
