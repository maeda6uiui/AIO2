import argparse
import hashlib
from pathlib import Path

def get_md5_hash(text:str)->str:
    return hashlib.md5(text.encode()).hexdigest()

def main(args):
    wikipedia_data_root_dirname:str=args.wikipedia_data_root_dirname
    targets_filepath:str=args.targets_filepath
    answer:str=args.answer

    wikipedia_data_root_dir=Path(wikipedia_data_root_dirname)

    target_titles=[]
    with open(targets_filepath,"r") as r:
        for line in r:
            title,score=line.split("\t")
            target_titles.append(title)

    target_hashes=[]
    for target_title in target_titles:
        target_hash=get_md5_hash(target_title)
        target_hashes.append(target_hash)

    for target_title,target_hash in zip(target_titles,target_hashes):
        wikipedia_data_dir=wikipedia_data_root_dir.joinpath(target_hash)
        text_file=wikipedia_data_dir.joinpath("text.txt")

        with text_file.open("r") as r:
            text=r.read()

        if answer in text:
            print(target_title)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--wikipedia_data_root_dirname",type=str,default="../Data/Wikipedia")
    parser.add_argument("--targets_filepath",type=str,default="../Data/Retriever/targets.txt")
    parser.add_argument("--answer",type=str)
    args=parser.parse_args()

    main(args)
