import argparse
import json

def main(args):
    input_filepath:str=args.input_filepath

    count=0
    with open(input_filepath,"r") as r:
        for line in r:
            data=json.loads(line)

            answer_ranges=data["answer_ranges"]
            count+=len(answer_ranges)

    print(count)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("-i","--input_filepath",type=str)
    args=parser.parse_args()

    main(args)
