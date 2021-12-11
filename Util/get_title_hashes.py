import argparse
import hashlib
from pathlib import Path

def get_md5_hash(text:str)->str:
    return hashlib.md5(text.encode()).hexdigest()

def main(args):
    input_filepath:str=args.input_filepath
    output_filepath:str=args.output_filepath

    input_file=Path(input_filepath)
    output_file=Path(output_filepath)

    with input_file.open("r") as r:
        titles=r.read().splitlines()

    title_hashes=[]
    for title in titles:
        title_hash=get_md5_hash(title)
        title_hashes.append(title_hash)

    title_hashes.sort()

    with output_file.open("w") as w:
        for title_hash in title_hashes:
            w.write("{}\n".format(title_hash))

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--input_filepath",type=str,default="../Data/wikipedia_titles.txt")
    parser.add_argument("--output_filepath",type=str,default="../Data/wikipedia_hashes.txt")
    args=parser.parse_args()
    
    main(args)
