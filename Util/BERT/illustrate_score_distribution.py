import argparse
import matplotlib.pyplot as plt

def main(args):
    input_filepath:str=args.input_filepath

    with open(input_filepath,"r") as r:
        lines=r.read().splitlines()
        lines=lines[2:]

    scores=[]
    for line in lines:
        score=line.split("\t")[1]
        score=float(score)

        scores.append(score)

    fig,ax=plt.subplots()
    ax.hist(scores)
    ax.set_xlabel("Score")
    ax.set_ylabel("Frequency")
    
    plt.show()

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--input_filepath",type=str,default="../Data/relevance_scores.txt")
    args=parser.parse_args()

    main(args)
