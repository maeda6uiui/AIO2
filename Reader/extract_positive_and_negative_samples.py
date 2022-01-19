import argparse
import json

def main(args):
    input_filepath:str=args.input_filepath
    positive_output_filepath:str=args.positive_output_filepath
    negative_output_filepath:str=args.negative_output_filepath

    positive_output_file=open(positive_output_filepath,"w")
    negative_output_file=open(negative_output_filepath,"w")

    with open(input_filepath,"r") as r:
        for line in r:
            data=json.loads(line)

            answer_ranges=data["answer_ranges"]
            if len(answer_ranges)>1:
                positive_output_file.write(line)
            else:
                if answer_ranges[0]!="0-0":
                    positive_output_file.write(line)
                else:
                    negative_output_file.write(line)

    positive_output_file.close()
    negative_output_file.close()

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("-i","--input_filepath",type=str)
    parser.add_argument("-p","--positive_output_filepath",type=str)
    parser.add_argument("-n","--negative_output_filepath",type=str)
    args=parser.parse_args()

    main(args)
