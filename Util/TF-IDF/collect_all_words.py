import argparse
import logging
from pathlib import Path
from tqdm import tqdm

logging_fmt="%(asctime)s %(levelname)s: %(message)s"
logging.basicConfig(format=logging_fmt)
logger=logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

def main(args):
    logger.info(args)

    wikipedia_data_root_dirname:str=args.wikipedia_data_root_dirname
    output_filepath:str=args.output_filepath

    logger.info("Wikipedia記事を読み込む準備をしています...")

    wikipedia_data_root_dir=Path(wikipedia_data_root_dirname)
    wikipedia_data_dirs=wikipedia_data_root_dir.glob("*")
    wikipedia_data_dirs=list(wikipedia_data_dirs)
    wikipedia_data_dirs.sort()

    logger.info("使用されているすべての単語を列挙します...")

    #ここで出力される単語の出現頻度=その単語が出現する文書の数
    #出力されたファイルに対して、
    #sort output.txt|uniq -c|sort -nr
    #とすることで、IDFの計算に必要なdf(t)を取得することができる
    with open(output_filepath,"w") as w:
        for wikipedia_data_dir in tqdm(wikipedia_data_dirs):
            genkei_frequencies_file:Path=wikipedia_data_dir.joinpath("genkei_frequencies.txt")
            with genkei_frequencies_file.open("r") as r:
                lines=r.read().splitlines()

            for line in lines:
                genkei,freq=line.split("\t")
                w.write("{}\n".format(genkei))

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--wikipedia_data_root_dirname",type=str,default="../../Data/Wikipedia")
    parser.add_argument("--output_filepath",type=str,default="../../Data/all_words.txt")
    args=parser.parse_args()

    main(args)
