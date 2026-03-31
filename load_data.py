import scanpy as sc
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch
import os
import anndata as ad
from scipy.sparse import issparse
from graph import graph_construction
import argparse
import warnings
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')


def load_data(args):
    if args.dataset == 'DLPFC':
        ## ---> check name of slice
        valid_names = ['151507', '151508', '151509', '151510', '151669', '151670', '151671', '151672', '151673', '151674', '151675', '151676']
        if args.slice_name not in valid_names:
            raise ValueError("Invalid slice name.")
        
        ## ---> load expression data
        adata = dlpfc_data_preprocess(args)
        args.adata = adata
        if args.slice_name in ['151669', '151670', '151671', '151672']:
            args.n_clusters = 5
        else:
            args.n_clusters = 7

        adj = graph_construction(adata, 20, 50, 'KNN')
        args.expression_adj = adj['adj_norm'].to(args.device)
        args.expression_adj = args.expression_adj.to_dense()
        args.expression_tensor = torch.tensor(adata.X.toarray(), dtype=torch.float32)
        args.expression_tensor = args.expression_tensor.to(args.device)

        ## ---> load tile data
        tile_path = f'./image_feature/DLPFC/{args.slice_name}/embeddings.npy'
        tile_np = np.load(tile_path)
        tile_tensor = torch.from_numpy(tile_np)
        args.tile_tensor = tile_tensor.to(args.device)

        print('n_clusters:', args.n_clusters)
        print('expression data shape:', args.expression_tensor.shape)
        print('expression adj shape:', args.expression_adj.shape)
        print('tile data shape:', args.tile_tensor.shape)
    
    elif args.dataset == 'AD':
        ## ---> check name of slice
        valid_names = ['2-5', 'T4857'] # 2-5 for health-control, T4857 for AD
        if args.slice_name not in valid_names:
            raise ValueError("Invalid slice name.")

        ## ---> load expression data
        adata, mask = ad_data_preprocess(args)
        args.adata = adata
        
        args.n_clusters = 6

        adj = graph_construction(adata, 20, 50, 'KNN')
        args.expression_adj = adj['adj_norm'].to(args.device)
        args.expression_adj = args.expression_adj.to_dense()
        args.expression_tensor = torch.tensor(adata.X, dtype=torch.float32)
        args.expression_tensor = args.expression_tensor.to(args.device)

        ## ---> load tile data
        tile_path = f'./image_feature/AD/{args.slice_name}/embeddings.npy'
        tile_np = np.load(tile_path)
        tile_tensor = torch.from_numpy(tile_np)
        args.tile_tensor = tile_tensor[mask].to(args.device)

        print('n_clusters:', args.n_clusters)
        print('expression data shape:', args.expression_tensor.shape)
        print('expression adj shape:', args.expression_adj.shape)
        print('tile data shape:', args.tile_tensor.shape)
    
    elif args.dataset == 'Mouse_brain':
        ## ---> check name of slice
        valid_names = 'ATAC'
        if args.slice_name not in valid_names:
            raise ValueError("Invalid slice name.")
        
        ## ---> load expression data
        adata = multi_omics_data_preprocess(args)
        args.adata = adata
        args.n_clusters = 12

        adj = graph_construction(adata, 20, 50, 'KNN') # use spatial data to construct graph
        args.expression_adj = adj['adj_norm'].to(args.device)
        args.expression_adj = args.expression_adj.to_dense()
        args.expression_tensor = torch.tensor(adata.X, dtype=torch.float32) # expression data is RNA
        args.expression_tensor = args.expression_tensor.to(args.device)

        ## ---> define ATAC data to take place of tile data
        tile_tensor = torch.tensor(adata.obsm['X_atac'], dtype=torch.float32)
        args.tile_tensor = tile_tensor.to(args.device) 

        print('n_clusters:', args.n_clusters)
        print('expression data shape:', args.expression_tensor.shape)
        print('expression adj shape:', args.expression_adj.shape)
        print('atac data shape:', args.tile_tensor.shape)

    elif args.dataset == 'Human_tonsil':
        ## ---> check name of slice
        valid_names = ['s2']
        if args.slice_name not in valid_names:
            raise ValueError("Invalid slice name.")
        
        ## ---> load expression data
        adata = tonsil_data_preprocess(args)
        args.adata = adata
        args.n_clusters = 4

        adj = graph_construction(adata, 12, 50, 'KNN')
        args.expression_adj = adj['adj_norm'].to(args.device)
        args.expression_adj = args.expression_adj.to_dense()
        args.expression_tensor = torch.tensor(adata.X, dtype=torch.float32) # expression data is RNA
        args.expression_tensor = args.expression_tensor.to(args.device)

        ## ---> define proteome data to take place of tile data
        tile_tensor = torch.tensor(adata.obsm['X_adt'], dtype=torch.float32)
        args.tile_tensor = tile_tensor.to(args.device) 

        print('n_clusters:', args.n_clusters)
        print('expression data shape:', args.expression_tensor.shape)
        print('expression adj shape:', args.expression_adj.shape)
        print('tile data shape:', args.tile_tensor.shape)
    
    elif args.dataset == 'chicken_heart':
        ## ---> check name of slice
        valid_names = ['D7', 'D10', 'D14']
        if args.slice_name not in valid_names:
            raise ValueError("Invalid slice name.")
        
        ## ---> load expression data
        adata = chicken_heart_data_preprocess(args)
        args.adata = adata
        if args.slice_name in ['D14']:
            args.n_clusters = 6
        else:
            args.n_clusters = 7

        adj = graph_construction(adata, 12, 50, 'KNN')
        args.expression_adj = adj['adj_norm'].to(args.device)
        args.expression_adj = args.expression_adj.to_dense()
        args.expression_tensor = torch.tensor(adata.X, dtype=torch.float32)
        args.expression_tensor = args.expression_tensor.to(args.device)

        ## ---> load tile data
        tile_path = f'./image_feature/chicken_heart/{args.slice_name}/embeddings.npy'
        tile_np = np.load(tile_path)
        tile_tensor = torch.from_numpy(tile_np)
        args.tile_tensor = tile_tensor.to(args.device)

        print('n_clusters:', args.n_clusters)
        print('expression data shape:', args.expression_tensor.shape)
        print('expression adj shape:', args.expression_adj.shape)
        print('tile data shape:', args.tile_tensor.shape)

    else:
        raise ValueError("Invalid dataset name.")


def dlpfc_data_preprocess(args):
    ## ---> load 
    expression_path = "./data/DLPFC/{}".format(args.slice_name)
    adata = sc.read_visium(expression_path)
    adata.var_names_make_unique()

    ## ---> preprocess
    adata.layers['count'] = adata.X.toarray()
    sc.pp.filter_genes(adata, min_cells=50)
    sc.pp.filter_genes(adata, min_counts=10)
    sc.pp.normalize_total(adata, target_sum=1e6)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", layer='count', n_top_genes=2000)
    adata = adata[:, adata.var['highly_variable'] == True]
    sc.pp.scale(adata)

    ## ---> add ground truth
    Ann_df = pd.read_csv(f"./data/DLPFC/{args.slice_name}/{args.slice_name}_truth.txt", sep='\t', header=None, index_col=0)
    Ann_df.columns = ['Ground Truth']
    adata.obs['ground_truth'] = Ann_df.loc[adata.obs_names, 'Ground Truth']
    adata = adata[~pd.isnull(adata.obs['ground_truth'])]

    return adata


def ad_data_preprocess(args):
    ## ---> load 
    expression_path = "./data/AD/{}/{}.h5ad".format(args.slice_name, args.slice_name)
    adata = sc.read(expression_path)
    adata.var_names_make_unique()

    ## ---> preprocess
    adata.layers['count'] = adata.X.toarray()
    sc.pp.filter_genes(adata, min_cells=50)
    sc.pp.filter_genes(adata, min_counts=10)
    sc.pp.normalize_total(adata, target_sum=1e6)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", layer='count', n_top_genes=2000)
    adata = adata[:, adata.var['highly_variable'] == True]
    sc.pp.scale(adata)

    ## ---> add ground truth
    mask = adata.obs['Layer'] != 'Noise'
    adata = adata[mask] # remove the noise
    adata.obs['ground_truth'] = adata.obs['Layer']
    adata = adata[~pd.isnull(adata.obs['ground_truth'])]

    return adata, mask


def multi_omics_data_preprocess(args):
    ## ---> load 
    omics_path = "./data/mouse_brain/E15.5-S1/E15_adata_atac_12.h5ad"
    adata = sc.read(omics_path)
    adata.var_names_make_unique()

    ## ---> preprocess RNA data
    adata.layers['count'] = adata.X.copy()
    sc.pp.filter_genes(adata, min_cells=50)
    sc.pp.filter_genes(adata, min_counts=10)
    sc.pp.normalize_total(adata, target_sum=1e6)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", layer='count', n_top_genes=2000)
    adata = adata[:, adata.var['highly_variable'] == True]
    sc.pp.scale(adata)

    ## ---> preprocess ATAC data
    x_atac = adata.obsm['X_atac']
    peak_names = adata.uns['atac_var_names']

    adata_atac = ad.AnnData(
    X=x_atac,
    obs=adata.obs.copy(),
    var=pd.DataFrame(index=peak_names),
    obsm={'spatial': adata.obsm['spatial']})

    sc.pp.normalize_total(adata_atac, target_sum=1e4)
    sc.pp.neighbors(adata_atac, use_rep='spatial')
    adata_atac = select_morani(adata_atac, nslt=2000)
    sc.pp.scale(adata_atac)
    
    adata.obsm['X_atac'] = adata_atac.X
    adata.uns['atac_var_names'] = adata_atac.var_names.tolist()
    
    # ---> add ground truth(don't have to modify)
    
    return adata


def tonsil_data_preprocess(args):
    ## ---> load
    omics_path = f"./data/Human_tonsil/combined_adata_{args.slice_name}.h5ad"
    adata = sc.read(omics_path)
    adata.var_names_make_unique()

    ## ---> preprocess RNA data
    adata.layers['count'] = adata.X.copy()
    sc.pp.filter_genes(adata, min_cells=50)
    sc.pp.filter_genes(adata, min_counts=10)
    sc.pp.normalize_total(adata, target_sum=1e6)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", layer='count', n_top_genes=2000)
    adata = adata[:, adata.var['highly_variable'] == True]
    sc.pp.scale(adata)

    # ---> preprocess ADT data
    adata.obsm['X_adt'] = adata.obsm['X_adt'].toarray()  # convert sparse matrix to dense
    tmp = sc.AnnData(adata.obsm['X_adt'])
    sc.pp.scale(tmp)
    adata.obsm['X_adt'] = tmp.X

    ## ---> add ground truth(remove NA)
    adata = adata[~pd.isnull(adata.obs['ground_truth'])]

    return adata


def chicken_heart_data_preprocess(args):
    ## ---> load 
    expression_path = f'./data/chicken_heart/{args.slice_name}/chicken_heart_{args.slice_name}.h5ad'
    adata = sc.read(expression_path)
    adata.var_names_make_unique()
    
    ## ---> preprocess
    adata.layers['count'] = adata.X.toarray()
    sc.pp.filter_genes(adata, min_cells=50)
    sc.pp.filter_genes(adata, min_counts=10)
    sc.pp.normalize_total(adata, target_sum=1e6)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", layer='count', n_top_genes=2000)
    adata = adata[:, adata.var['highly_variable'] == True]
    sc.pp.scale(adata)

    ## ---> add ground truth(don't have to modify)

    return adata

def select_morani(adata, nslt=1000, morans_method='scanpy'):  
        if morans_method == 'scanpy':  
            print('Computing Moran\'s I...')  
            morani = sc.metrics.morans_i(adata)  
            m_order = np.flip(np.argsort(morani))  
            
            slt_m = m_order[0:nslt]  
            adata = adata[:, slt_m]  
            if issparse(adata.X):  
                exp_mat = adata.X.toarray().astype(np.float32)
            else:  
                exp_mat = adata.X.astype(np.float32)
            print('Finished gene selection')
            return adata
        else:
            raise ValueError("Unknown morans_method")
    