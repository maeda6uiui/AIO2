import argparse
import logging
import math

logging_fmt="%(asctime)s %(levelname)s: %(message)s"
logging.basicConfig(format=logging_fmt)
logger=logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

def main(args):
    logger.info(args)

    total_num_documents:int=args.total_num_documents
    dft_filepath:str=args.dft_filepath
    output_filepath:str=args.output_filepath

    logger.info("IDFの計算を行っています...")

    with open(output_filepath,"w") as w:
        with open(dft_filepath,"r") as r:
            for line in r:
                if line=="":
                    continue

                line=line.strip()
                freq,genkei=line.split(" ")
                freq=int(freq)

                idf=math.log(total_num_documents/freq)+1

                w.write("{}\t{}\n".format(genkei,idf))

    logger.info("処理が完了しました")

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--total_num_documents",type=int,default=920172)
    parser.add_argument("--dft_filepath",type=str,default="../../Data/dft.txt")
    parser.add_argument("--output_filepath",type=str,default="../../Data/idf.txt")
    args=parser.parse_args()

    main(args)
