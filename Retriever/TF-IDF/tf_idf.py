import logging
import pickle
import re
import subprocess
from pathlib import Path
from scipy import sparse
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
        self.vectorizer=TfidfVectorizer(token_pattern=u'(?u)\\b\\w+\\b')

    def load_vectorizer(self,vectorizer_filepath:str):
        self.vectorizer:TfidfVectorizer=pickle.load(open(vectorizer_filepath,"rb"))

    def save_vectorizer(self,vectorizer_filepath:str):
        pickle.dump(self.vectorizer,open(vectorizer_filepath,"wb"))

    def fit_transform_corpus(self,wikipedia_data_dirs:List[Path],matrix_save_file:Path=None):
        genkei_filepaths:List[str]=[]
        for wikipedia_data_dir in wikipedia_data_dirs:
            genkei_file=wikipedia_data_dir.joinpath("genkeis.txt")
            genkei_filepaths.append(str(genkei_file))

        self.vectorizer.set_params(input="filename")

        self.vectorizer=self.vectorizer.fit(genkei_filepaths)
        tf_idf_matrix=self.vectorizer.transform(genkei_filepaths)

        self.vectorizer.set_params(input="content")

        if matrix_save_file is not None:
            sparse.save_npz(matrix_save_file,tf_idf_matrix)

        return tf_idf_matrix

    def transform_corpus(self,wikipedia_data_dirs:List[Path]):
        genkei_filepaths:List[str]=[]
        for wikipedia_data_dir in wikipedia_data_dirs:
            genkei_file=wikipedia_data_dir.joinpath("genkeis.txt")
            genkei_filepaths.append(str(genkei_file))

        self.vectorizer.set_params(input="filename")
        tf_idf_matrix=self.vectorizer.transform(genkei_filepaths)
        self.vectorizer.set_params(input="content")

        return tf_idf_matrix

    def transform_query(self,query:str):
        #半角スペースで分かち書きされたクエリを入力する
        tf_idf_matrix=self.vectorizer.transform([query])
        return tf_idf_matrix
