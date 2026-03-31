import scanpy as sc
import numpy as np
import pandas as pd
import os
from anndata import AnnData


class LoadSingle10xAdata:
    def __init__(self, path: str, n_top_genes: int = 3000, n_neighbors: int = 3, image_emb: bool = False, label: bool = True, filter_na: bool = True,select = 'default', slice_name = '151507'):
        self.path = path
        self.n_top_genes = n_top_genes
        self.n_neighbors = n_neighbors
        self.adata = None
        self.image_emb = image_emb
        self.label = label
        self.filter_na = filter_na
        self.kernel = 'euclidean'
        self.select = 'default'
        self.slice_name = slice_name


    def load_data(self):
        self.adata = sc.read_visium(self.path, count_file='filtered_feature_bc_matrix.h5', load_images=True)
        self.adata.var_names_make_unique()


    def load_label(self):
        df_meta = pd.read_csv(os.path.join(self.path, f'{self.slice_name}_truth.txt'), sep='\t', header=None)
        df_meta_layer = df_meta[1]

        self.adata.obs['ground_truth'] = df_meta_layer.values

        if self.filter_na:
            self.adata = self.adata[~pd.isnull(self.adata.obs['ground_truth'])]

