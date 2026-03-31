import os
import numpy as np
from sklearn.cluster import KMeans
import scanpy as sc
import matplotlib.pyplot as plt
from sklearn import metrics
import pandas as pd
from scipy.spatial import distance

def clustering_kmeans(adata, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, n_init=40, random_state=42)
    kmeans.fit(adata.obsm["best_emb"])
    
    adata.obs["STransformer_domain"] = kmeans.labels_
    adata.obs['STransformer_domain'] = adata.obs['STransformer_domain'].astype('category')
    
    return adata


def plot_clustering(adata, n_clusters, img_save_path, dataset):
    
    adata = clustering_kmeans(adata, n_clusters) # kmeans

    adj_2d = distance.cdist(adata.obsm['spatial'], adata.obsm['spatial'], 'euclidean')
    if dataset in ['DLPFC', 'AD', 'Human_tonsil', 'chicken_heart']:
        refined_pred = refine(sample_id=adata.obs.index.tolist(), 
                            pred=adata.obs["STransformer_domain"].tolist(), 
                            dis=adj_2d)
    elif dataset == 'Mouse_brain':
        refined_pred = refine(sample_id=adata.obs.index.tolist(), 
                            pred=adata.obs["STransformer_domain"].tolist(), 
                            dis=adj_2d, shape='square')
    else:
        raise ValueError("Invalid dataset name.")    

    # add refine domain
    adata.obs['refine_STransformer_domain'] = adata.obs['STransformer_domain'].copy()
    adata.obs['refine_STransformer_domain'].values[:] = refined_pred


    if dataset in ['DLPFC', 'AD', 'Human_tonsil', 'chicken_heart']:
        ARI = metrics.adjusted_rand_score(adata.obs['ground_truth'], adata.obs['refine_STransformer_domain'])
        print(f'The final ARI is {ARI:.4f}.')
    
    color = ['#D5695D', '#F5B041', '#F6DA65', '#52BE80', '#91DFD0', '#5DADE2', '#A469BD', '#484F98']
    if dataset == 'DLPFC':
        sc.pl.spatial(adata, color='refine_STransformer_domain', frameon = False, spot_size=150, palette=color, img_key=None)
    elif dataset == 'AD':
        sc.pl.spatial(adata, color='refine_STransformer_domain', frameon = False, spot_size=25, palette=color, img_key=None)
    elif dataset == 'Mouse_brain':
        color = ['#D5695D', '#F5B041', '#F6DA65', '#52BE80', '#91DFD0', '#5DADE2', '#A469BD', '#484F98', "#748E84", "#E1DD10", "#A36091", "#5E2633"]
        sc.pl.spatial(adata, color='refine_STransformer_domain', frameon = False, spot_size=1, palette=color)
    elif dataset == 'Human_tonsil':
        sc.pl.spatial(adata, color='refine_STransformer_domain', frameon = False, spot_size=1.8, palette=color)
    elif dataset == 'chicken_heart':
        sc.pl.spatial(adata, color='refine_STransformer_domain', frameon = False, spot_size=300, palette=color, img_key=None)
    
    if dataset in ['DLPFC', 'AD', 'Human_tonsil', 'chicken_heart']:
        plt.title('STransformer \n (ARI=%.4f)' % ARI)
    
    plt.savefig(img_save_path, bbox_inches='tight', dpi=300)


def refine(sample_id, pred, dis, shape="hexagon"):
    refined_pred=[]
    pred=pd.DataFrame({"pred": pred}, index=sample_id)
    dis_df=pd.DataFrame(dis, index=sample_id, columns=sample_id)
    if shape=="hexagon":
        num_nbs=6
    elif shape=="square":
        num_nbs=4
    else:
        print("Shape not recongized, shape='hexagon' for Visium data.")
    for i in range(len(sample_id)):
        index=sample_id[i]
        dis_tmp=dis_df.loc[index, :].sort_values()
        nbs=dis_tmp[0:num_nbs+1]
        nbs_pred=pred.loc[nbs.index, "pred"]
        self_pred=pred.loc[index, "pred"]
        v_c=nbs_pred.value_counts()
        if (v_c.loc[self_pred]<num_nbs/2) and (np.max(v_c)>num_nbs/2):
            refined_pred.append(v_c.idxmax())
        else:           
            refined_pred.append(self_pred)
    return refined_pred