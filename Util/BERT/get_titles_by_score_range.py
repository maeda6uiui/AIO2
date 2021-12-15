import argparse

def main(args):
    input_filepath:str=args.input_filepath
    output_filepath:str=args.output_filepath
    score_lower_bound:float=args.score_lower_bound
    score_upper_bound:float=args.score_upper_bound

    with open(input_filepath,"r") as r:
        lines=r.read().splitlines()

    input_text=lines[0]
    lines=lines[2:]

    titles=[]
    for line in lines:
        title,score=line.split("\t")
        score=float(score)
        if score>=score_lower_bound and score<=score_upper_bound:
            titles.append(title)

    with open(output_filepath,"w") as w:
        w.write("入力テキスト: {}\n\n".format(input_text))

        for title in titles:
            w.write("{}\n".format(title))

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--input_filepath",type=str,default="../Data/relevance_scores.txt")
    parser.add_argument("--output_filepath",type=str,default="../Data/titles_by_score_range.txt")
    parser.add_argument("--score_lower_bound",type=float,default=0.90)
    parser.add_argument("--score_upper_bound",type=float,default=0.99)
    args=parser.parse_args()

    main(args)
