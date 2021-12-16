import logging
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
        buffer_size:int):
        self.mecab_args=["mecab","-b",str(buffer_size),"-d",mecab_dictionary_dirname]

        self.r1=re.compile(r"https?://[\w/:%#\$&\?\(\)~\.=\+\-…]+")
        self.r2=re.compile(r"[!-~]")
        self.r3=re.compile(r"[：-＠]")

    def extract_genkeis(self,text_file:Path)->List[str]:
        this_mecab_args=self.mecab_args.copy()
        this_mecab_args.append(str(text_file))

        proc=subprocess.run(this_mecab_args,stdout=subprocess.PIPE)
        ma_result=proc.stdout.decode("utf8")
        lines=ma_result.splitlines()
        lines.pop()

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
    def __init__(self):
        self.tf_idf_vectorizer=TfidfVectorizer(
            analyzer="word",
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
        self.tf_idf_vectorizer.set_params(input="content")

    def save_tf_idf_vectorizer(self,vectorizer_filepath:str):
        pickle.dump(self.tf_idf_vectorizer,open(vectorizer_filepath,"wb"))

    def transform(self,text:str):
        tf_idf_matrix=self.tf_idf_vectorizer.transform([text])
        return tf_idf_matrix
