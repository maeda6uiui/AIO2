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
    titles_save_filepath:str=args.titles_save_filepath

    logger.info("Wikipediaデータを読み込む準備をしています...")

    wikipedia_data_root_dir=Path(wikipedia_data_root_dirname)
    titles_save_file=Path(titles_save_filepath)

    wikipedia_data_dirs=wikipedia_data_root_dir.glob("*")
    wikipedia_data_dirs=list(wikipedia_data_dirs)

    logger.info("データ数: {}".format(len(wikipedia_data_dirs)))

    logger.info("Wikipedia記事のタイトルを取得しています...")

    with titles_save_file.open("w") as w:
        for wikipedia_data_dir in tqdm(wikipedia_data_dirs):
            title_file=wikipedia_data_dir.joinpath("title.txt")

            with title_file.open("r") as r:
                title=r.read().splitlines()[0]
                
                w.write(title)
                w.write("\n")

    logger.info("処理が完了しました")

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--wikipedia_data_root_dirname",type=str,default="../Data/Wikipedia")
    parser.add_argument("--titles_save_filepath",type=str,default="../Data/wikipedia_titles.txt")
    args=parser.parse_args()

    main(args)
