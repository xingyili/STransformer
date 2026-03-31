import torch
import torch.optim as optim
import pandas as pd
import numpy as np
import argparse
import copy
import os
from sklearn.cluster import KMeans
from sklearn import metrics
from timeit import default_timer as timer
from load_data import load_data
from models.model import STransformer
from loss import calculate_loss
from plot import plot_clustering
from datetime import datetime
from tqdm import tqdm


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_args():
    parser = argparse.ArgumentParser(description='Configurations for STransformer project')
    parser.add_argument('--dataset', type=str, help='dataset name', default='DLPFC')
    parser.add_argument('--slice_name', type=str, help='slice name', default='151507')
    parser.add_argument('--t_epoch', type=int, help='training epoch', default=500)
    parser.add_argument('--lr', type=float, help='learning rate', default=0.00005)
    parser.add_argument('--drop_feature_rate', type=float, help='drop feature rate', default=0.05)
    parser.add_argument('--drop_edge_rate', type=float, help='drop edge rate', default=0.05)
    parser.add_argument('--cuda', type=str, help='cuda device', default='0')
    args = parser.parse_args()
    return args

def init_model(args):
    if args.dataset == 'DLPFC':
        n = 3 if args.slice_name in ['151673', '151674', '151675', '151676'] else 2
        model = STransformer(N = n, 
                         d_token = 256, 
                         d_emd = 128, 
                         d_input = args.expression_tensor.shape[1], 
                         d_output = 64
                         ).to(args.device)
    else:
        print('dataset not supported yet')
        model = None
    
    return model

def calculate_ari(ground_truth, domain):
    ari = metrics.adjusted_rand_score(ground_truth, domain)
    return ari

def get_kmeans(z_tilde, adata, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, n_init=40, random_state=42)
    z_tilde = z_tilde.data.cpu().numpy()
    kmeans.fit(z_tilde)
    adata.obs["STransformer_domain"] = kmeans.labels_
    adata.obs['STransformer_domain'] = adata.obs['STransformer_domain'].astype('category')
    ari = calculate_ari(adata.obs['ground_truth'], adata.obs['STransformer_domain'])
    return ari


def train(model, optimizer, args):
    # save path
    save_dir = f"./save/{args.dataset}/{args.slice_name}"
    os.makedirs(save_dir, exist_ok=True)

    # preparation
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=80, gamma=0.5)
    best_ari = 0.0
    best_epoch = 0
    best_embedding = None
    best_recon_expression = None
    best_recon_tile = None

    # start training
    for epoch in tqdm(range(args.t_epoch), desc="Training Epochs"):
        model.train()
        mm_embed, recon_expression, recon_tile = model(args.tile_tensor, args.expression_tensor, args.expression_adj)
        recon_loss_exp, recon_loss_tile, cosine_loss_exp, cosine_loss_tile = calculate_loss(args.expression_tensor, recon_expression, args.tile_tensor.float(), recon_tile)
        loss = 0.45*recon_loss_exp + 0.45*recon_loss_tile + 0.05*cosine_loss_exp + 0.05*cosine_loss_tile
        ari = get_kmeans(mm_embed, args.adata, args.n_clusters) # kmeans
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if ari >= best_ari:
            best_ari = ari
            best_epoch = epoch
            best_embedding = mm_embed
            best_recon_expression = recon_expression
            best_recon_tile = recon_tile

    args.adata.obsm["best_emb"] = best_embedding.cpu().detach().numpy()
    
    # save reconstructed data
    args.adata.obsm['denoise_X'] = best_recon_expression.cpu().detach().numpy()
    args.adata.obsm['denoise_Y'] = best_recon_tile.cpu().detach().numpy()

    # plot
    img_save_path = os.path.join(save_dir, f"{args.slice_name}_clustering_results.pdf")
    plot_clustering(args.adata, args.n_clusters, img_save_path, args.dataset)
    
    # save results
    args.adata.write(os.path.join(save_dir, f'{args.slice_name}_results.h5ad'), compression="gzip")


def main():
    args = get_args()
    print('======Environment Setup======')
    if args.cuda != 'cpu':
        print('Device: cuda '+ args.cuda)
        args.device = torch.device('cuda:'+ args.cuda)
    else:
        args.device = torch.device('Device cpu')
    
    print('========Data Loading========')
    if args.dataset in ['DLPFC','AD', 'Mouse_brain', 'Human_tonsil', 'chicken_heart']:
        load_data(args)
    else:
        raise ValueError("Invalid dataset name.")
    
    print('=====Model Initializing=====')
    model = init_model(args)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    print('=======Training Phase=======')
    train(model, optimizer, args)

    print('========Project Done========')
    
    print('=====Save Successfully!=====')


if __name__ == '__main__':
    set_seed(42)
    # start = timer()
    main()
    # end = timer()
    # print('Script Time: %f seconds' % (end - start))