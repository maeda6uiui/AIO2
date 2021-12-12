import argparse
from pathlib import Path
from tqdm import tqdm

def main(args):
    vectors_dirname:str=args.vectors_dirname
    wikipedia_data_root_dirname:str=args.wikipedia_data_root_dirname

    vectors_dir=Path(vectors_dirname)
    wikipedia_data_root_dir=Path(wikipedia_data_root_dirname)

    vector_files=vectors_dir.glob("*.pt")
    vector_files=list(vector_files)

    for vector_file in tqdm(vector_files):
        title_hash=vector_file.stem
        wikipedia_data_dir=wikipedia_data_root_dir.joinpath(title_hash)

        dst_file=wikipedia_data_dir.joinpath("vector.pt")

        vector_file.rename(dst_file)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--vectors_dirname",type=str,default="../Data/WikipediaVector")
    parser.add_argument("--wikipedia_data_root_dirname",type=str,default="../Data/Wikipedia")
    args=parser.parse_args()

    main(args)
