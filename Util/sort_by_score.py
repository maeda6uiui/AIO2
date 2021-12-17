import argparse

def main(args):
    input_filepath:str=args.input_filepath
    output_filepath:str=args.output_filepath

    with open(input_filepath,"r") as r:
        lines=r.read().splitlines()

    records={}
    for line in lines:
        title,score=line.split("\t")
        score=float(score)

        records[title]=score

    sorted_records=dict(sorted(records.items(),key=lambda item:item[1],reverse=True))

    with open(output_filepath,"w") as w:
        for title,score in sorted_records.items():
            w.write("{}\t{}\n".format(title,score))

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--input_filepath",type=str,default="../Data/Retriever/relevance_scores_mul.txt")
    parser.add_argument("--output_filepath",type=str,default="../Data/Retriever/relevance_scores_mul_sorted.txt")
    args=parser.parse_args()

    main(args)
