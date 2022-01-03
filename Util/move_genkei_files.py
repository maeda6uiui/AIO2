import argparse
import logging
import shutil
from pathlib import Path
from tqdm import tqdm

logging_fmt="%(asctime)s %(levelname)s: %(message)s"
logging.basicConfig(format=logging_fmt)
logger=logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

def main(args):
    logger.info(args)

    wikipedia_data_root_dirname:str=args.wikipedia_data_root_dirname
    output_root_dirname:str=args.output_root_dirname

    wikipedia_data_root_dir=Path(wikipedia_data_root_dirname)
    wikipedia_data_dirs=wikipedia_data_root_dir.glob("*")
    wikipedia_data_dirs=list(wikipedia_data_dirs)
    wikipedia_data_dirs.sort()

    output_root_dir=Path(output_root_dirname)
    output_root_dir.mkdir(parents=True,exist_ok=True)

    for wikipedia_data_dir in tqdm(wikipedia_data_dirs):
        title_hash=wikipedia_data_dir.name
        output_dir=output_root_dir.joinpath(title_hash)
        output_dir.mkdir(exist_ok=True)
        
        genkei_src_file:Path=wikipedia_data_dir.joinpath("genkeis.txt")
        genkei_dst_file=output_dir.joinpath("genkeis.txt")
        genkei_src_file.rename(genkei_dst_file)

        title_file=wikipedia_data_dir.joinpath("title.txt")
        shutil.copy(title_file,output_dir)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--wikipedia_data_root_dirname",type=str,default="../Data/Wikipedia")
    parser.add_argument("--output_root_dirname",type=str,default="../Data/WakatiWithMecab")
    args=parser.parse_args()
    
    main(args)
