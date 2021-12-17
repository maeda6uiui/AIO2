import argparse
import hashlib

def get_md5_hash(text:str)->str:
    return hashlib.md5(text.encode()).hexdigest()

def main(args):
    input_text:str=args.input_text

    md5_hash=get_md5_hash(input_text)
    print("{}\t{}".format(input_text,md5_hash))

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("-i","--input_text",type=str)
    args=parser.parse_args()

    main(args)
