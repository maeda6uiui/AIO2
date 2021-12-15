import argparse

def main(args):
    input_filepath:str=args.input_filepath
    score_lower_bound:float=args.score_lower_bound
    score_upper_bound:float=args.score_upper_bound

    with open(input_filepath,"r") as r:
        lines=r.read().splitlines()

    lines=lines[2:]

    titles=[]
    for line in lines:
        title,score=line.split("\t")
        score=float(score)
        if score>=score_lower_bound and score<=score_upper_bound:
            titles.append(title)

    print(len(titles))

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--input_filepath",type=str,default="../Data/relevance_scores.txt")
    parser.add_argument("--score_lower_bound",type=float,default=0.90)
    parser.add_argument("--score_upper_bound",type=float,default=0.99)
    args=parser.parse_args()

    main(args)
