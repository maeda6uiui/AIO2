import argparse
import hashlib
import logging
import torch
from transformers import AutoTokenizer,BertConfig,BertModel
from pathlib import Path
from tqdm import tqdm

logging_fmt="%(asctime)s %(levelname)s: %(message)s"
logging.basicConfig(format=logging_fmt)
logger=logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DocumentVectorCreator(object):
    def __init__(self,bert_model_name:str):
        self.bert=BertModel.from_pretrained(bert_model_name)
        self.bert.eval()
        self.bert.to(device)

        self.tokenizer=AutoTokenizer.from_pretrained(bert_model_name)

        config=BertConfig.from_pretrained(bert_model_name)
        self.max_length=config.max_position_embeddings

    def create_document_vector(self,text:str)->torch.FloatTensor:
        inputs=self.tokenizer.encode_plus(
            text,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt")

        bert_inputs={
            "input_ids":inputs["input_ids"].to(device),
            "attention_mask":inputs["attention_mask"].to(device),
            "token_type_ids":inputs["token_type_ids"].to(device),
            "return_dict":True
        }
        
        with torch.no_grad():
            outputs=self.bert(**bert_inputs)
            pooler_output=outputs["pooler_output"]

        return pooler_output.cpu()

def get_md5_hash(text:str)->str:
    return hashlib.md5(text.encode()).hexdigest()

def main(args):
    logger.info(args)

    wikipedia_data_root_dirname:str=args.wikipedia_data_root_dirname
    document_vectors_save_dirname:str=args.document_vectors_save_dirname
    bert_model_name:str=args.bert_model_name

    logger.info("Wikipediaデータを読み込む準備をしています...")

    wikipedia_data_root_dir=Path(wikipedia_data_root_dirname)
    wikipedia_data_dirs=wikipedia_data_root_dir.glob("*")
    wikipedia_data_dirs=list(wikipedia_data_dirs)

    logger.info("Wikipediaデータの数: {}".format(len(wikipedia_data_dirs)))

    logger.info("文書ベクトルの作成を行っています...")

    document_vectors_save_dir=Path(document_vectors_save_dirname)
    document_vectors_save_dir.mkdir(parents=True,exist_ok=True)

    vec_creator=DocumentVectorCreator(bert_model_name)

    for wikipedia_data_dir in tqdm(wikipedia_data_dirs):
        title_file:Path=wikipedia_data_dir.joinpath("title.txt")
        text_file:Path=wikipedia_data_dir.joinpath("text.txt")

        with title_file.open("r") as r:
            title=r.read().splitlines()[0]

        with text_file.open("r") as r:
            text=r.read().splitlines()[0]

        document_vector=vec_creator.create_document_vector(text)

        title_hash=get_md5_hash(title)
        document_vector_save_file=document_vectors_save_dir.joinpath("{}.pt".format(title_hash))

        torch.save(document_vector,document_vector_save_file)

    logger.info("処理が完了しました")

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--wikipedia_data_root_dirname",type=str,default="../Data/Wikipedia")
    parser.add_argument("--document_vectors_save_dirname",type=str,default="../Data/WikipediaVector")
    parser.add_argument("--bert_model_name",type=str,default="cl-tohoku/bert-base-japanese-whole-word-masking")
    args=parser.parse_args()

    main(args)
