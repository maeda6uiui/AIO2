import argparse
import logging
import numpy as np
import pickle
import re
import subprocess
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List

logging_fmt="%(asctime)s %(levelname)s: %(message)s"
logging.basicConfig(format=logging_fmt)
logger=logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

class GenkeiExtractor(object):
    def __init__(
        self,
        mecab_dictionary_dirname:str,
        buffer_size:int,
        working_filepath:str):
        self.mecab_args=["mecab","-b",str(buffer_size),"-d",mecab_dictionary_dirname]

        self.r1=re.compile(r"https?://[\w/:%#\$&\?\(\)~\.=\+\-…]+")
        self.r2=re.compile(r"[!-~]")
        self.r3=re.compile(r"[：-＠]")

        self.working_file=Path(working_filepath)

    def extract_genkeis(self,text:str)->List[str]:
        with self.working_file.open("w") as w:
            w.write(text)

        this_mecab_args=self.mecab_args.copy()
        this_mecab_args.append(str(self.working_file))

        proc=subprocess.run(this_mecab_args,stdout=subprocess.PIPE)
        ma_result=proc.stdout.decode("utf8")
        lines=ma_result.splitlines()
        lines.pop()

        self.working_file.unlink()

        genkeis=[]
        for line in lines:
            details=line.split("\t")
            if len(details)!=2:
                continue

            word=details[0]
            if self.r1.match(word) is not None:
                continue
            if self.r2.match(word) is not None:
                continue
            if self.r3.match(word) is not None:
                continue

            genkei=details[1].split(",")[6]
            genkeis.append(genkei)

        return genkeis

class TFIDFCalculator(object):
    def __init__(
        self,
        mecab_dictionary_dirname:str,
        buffer_size:int,
        working_filepath:str):
        self.genkei_extractor=GenkeiExtractor(mecab_dictionary_dirname,buffer_size,working_filepath)

        self.tf_idf_vectorizer=TfidfVectorizer(
            analyzer=self.genkei_extractor.extract_genkeis,
            use_idf=True,
            norm="l2",
            smooth_idf=True,
            input="filename")

    def fit_tf_idf_vectorizer(self,wikipedia_data_dirs:List[Path]):
        text_filepaths:List[str]=[]
        for wikipedia_data_dir in wikipedia_data_dirs:
            text_file=wikipedia_data_dir.joinpath("text.txt")
            text_filepaths.append(str(text_file))

        self.tf_idf_vectorizer.fit(text_filepaths)

    def load_tf_idf_vectorizer(self,vectorizer_filepath:str):
        self.tf_idf_vectorizer:TfidfVectorizer=pickle.load(open(vectorizer_filepath,"rb"))
        self.tf_idf_vectorizer.set_params(
            analyzer=self.genkei_extractor.extract_genkeis,
            input="content")

    def save_tf_idf_vectorizer(self,vectorizer_filepath:str):
        self.tf_idf_vectorizer.set_params(analyzer="word")
        pickle.dump(self.tf_idf_vectorizer,open(vectorizer_filepath,"wb"))

    def transform(self,text:str):
        tf_idf_matrix=self.tf_idf_vectorizer.transform([text])
        return tf_idf_matrix

def main(args):
    logger.info(args)

    wikipedia_data_root_dirname:str=args.wikipedia_data_root_dirname
    mecab_dictionary_dirname:str=args.mecab_dictionary_dirname
    buffer_size:int=args.buffer_size
    start_index:int=args.start_index
    end_index:int=args.end_index
    vectorizer_save_filepath:str=args.vectorizer_save_filepath
    working_filepath:str=args.working_filepath

    logger.info("Vectorizerを学習する準備をしています...")

    wikipedia_data_root_dir=Path(wikipedia_data_root_dirname)
    wikipedia_data_dirs=wikipedia_data_root_dir.glob("*")
    wikipedia_data_dirs=list(wikipedia_data_dirs)
    wikipedia_data_dirs.sort()

    logger.info("Wikipedia記事数: {}".format(len(wikipedia_data_dirs)))

    if end_index is None:
        end_index=len(wikipedia_data_dirs)

    logger.info("{}から{}のWikipedia記事に対して処理を行います".format(start_index,end_index))

    wikipedia_data_dirs=wikipedia_data_dirs[start_index:end_index]

    tf_idf_calculator=TFIDFCalculator(mecab_dictionary_dirname,buffer_size,working_filepath)

    logger.info("Vectorizerの学習を行っています...")

    tf_idf_calculator.fit_tf_idf_vectorizer(wikipedia_data_dirs)
    tf_idf_calculator.save_tf_idf_vectorizer(vectorizer_save_filepath)

    logger.info("処理が完了しました")

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--wikipedia_data_root_dirname",type=str,default="../Data/Wikipedia")
    parser.add_argument("--mecab_dictionary_dirname",type=str,default="/usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd")
    parser.add_argument("--buffer_size",type=int,default=1024*1024*10)
    parser.add_argument("--start_index",type=int,default=0)
    parser.add_argument("--end_index",type=int)
    parser.add_argument("--vectorizer_save_filepath",type=str,default="../Data/tf_idf_vectorizer.pkl")
    parser.add_argument("--working_filepath",type=str,default="../Data/working_file.txt")
    args=parser.parse_args()

    main(args)
