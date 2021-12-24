import argparse
from pathlib import Path

def main(args):
    output_filepath:str=args.output_filepath

    output_file=Path(output_filepath)
    output_dir=output_file.parent

    if not output_dir.exists():
        output_dir.mkdir(parents=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("-o","--output_filepath",type=str)
    args=parser.parse_args()

    main(args)
