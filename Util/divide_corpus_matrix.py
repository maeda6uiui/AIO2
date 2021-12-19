import argparse
import numpy as np
import torch
from pathlib import Path
from scipy import sparse

def convert_csr_to_tensor(csr:sparse.csr_matrix)->torch.FloatTensor:
    coo=csr.tocoo()

    values=coo.data
    indices=np.vstack((coo.row,coo.col))

    i=torch.LongTensor(indices)
    v=torch.FloatTensor(values)
    shape=coo.shape

    tensor=torch.sparse_coo_tensor(i,v,torch.Size(shape))

    return tensor

def main(args):
    corpus_matrix_filepath:str=args.corpus_matrix_filepath
    corpus_tensors_output_dirname:str=args.corpus_tensors_output_dirname
    num_divisions:int=args.num_divisions

    corpus_tensors_output_dir=Path(corpus_tensors_output_dirname)
    corpus_tensors_output_dir.mkdir(parents=True,exist_ok=True)

    corpus_matrix:sparse.csr_matrix=sparse.load_npz(corpus_matrix_filepath)
    num_rows=corpus_matrix.shape[0]

    for i in range(num_divisions):
        start_row=int(num_rows/num_divisions)*i
        end_row=int(num_rows/num_divisions)*(i+1)

        csr=None
        if i==num_divisions-1:
            csr=corpus_matrix[start_row:,:]
        else:
            csr=corpus_matrix[start_row:end_row,:]

        tensor=convert_csr_to_tensor(csr).coalesce()

        indices_file=corpus_tensors_output_dir.joinpath("indices_{}.pt".format(i))
        values_file=corpus_tensors_output_dir.joinpath("values_{}.pt".format(i))
        size_file=corpus_tensors_output_dir.joinpath("size_{}.pt".format(i))
        
        torch.save(tensor.indices(),indices_file)
        torch.save(tensor.values(),values_file)
        torch.save(tensor.size(),size_file)

    info_file=corpus_tensors_output_dir.joinpath("info.txt")
    with info_file.open("w") as w:
        w.write("num_divisions={}\n".format(num_divisions))

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--corpus_matrix_filepath",type=str,default="../Data/Retriever/TF-IDF/corpus_matrix.npz")
    parser.add_argument("--corpus_tensors_output_dirname",type=str,default="../Data/Retriever/TF-IDF/CorpusTensor")
    parser.add_argument("--num_divisions",type=int,default=8)
    args=parser.parse_args()

    main(args)
