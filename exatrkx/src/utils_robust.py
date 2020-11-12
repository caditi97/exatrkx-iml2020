import sys
import os
import torch
import pytorch_lightning as pl
import yaml
import importlib
import trackml
import trackml.dataset
import numpy as np

from torch_cluster import radius_graph
from utils_torch import build_edges
from embedding.layerless_embedding import *
from embedding.embedding_base import *
# from src.filter import filter_base
# from src.filter.vanilla_filter import *

import matplotlib.pyplot as plt

#default values
os.environ['TRKXINPUTDIR']="/global/cfs/cdirs/m3443/data/trackml-kaggle/train_10evts"
os.environ['TRKXOUTPUTDIR']= "/global/cfs/projectdirs/m3443/usr/caditi97/iml2020/out0"


#############################################
#                  PLOTS                    #
#############################################


def plot_noise_dist(dir_path, noise_keeps):
    noise = []
    not_noise = []
    for i in noise_keeps:
        data_path = f"/out{i}/feature_store/1000"
        #data = torch.load(f"/global/cfs/projectdirs/m3443/usr/caditi97/iml2020/feature_store_endcaps/n{i}/1000")
        data = torch.load(dir_path + data_path)
        arr = data['pid']
        n_count = np.count_nonzero(arr==0)
        not_n = np.count_nonzero(arr)
        noise.append(n_count)
        not_noise.append(not_n)
        print("-----")
        print(data)

    x = np.arange(len(noise_keeps))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(10,5))
    rects1 = ax.bar(x - width/2, noise, width, label='noise')
    rects2 = ax.bar(x + width/2, not_noise, width, label='not noise')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('# of hits')
    ax.set_xlabel('keep')
    ax.set_xticks(x)
    labels = noise_keeps
    ax.set_xticklabels(labels)
    ax.legend()
    
    
#############################################
#                 EMBEDDING                 #
#############################################

def get_emb_ckpt(emb_ckpt_path, train_split=None, cluster=None):
    emb_ckpt = torch.load(emb_ckpt_path)
    if train_split is not None:
        emb_ckpt['hyper_parameters']['train_split'] = train_split
    if cluster is not None:
        emb_ckpt['hyper_parameters']['clustering'] = cluster
    return emb_ckpt


def load_cktp(ckpt, ckpt_path, emb=False, filtering=False):
    if emb:
        emb_model = LayerlessEmbedding(ckpt['hyper_parameters'])
        best_model = emb_model.load_from_checkpoint(ckpt_path, hparams=ckpt['hyper_parameters'])
    #add filtering stuff
    return best_model


def get_cluster(best_emb,batch):
    if 'ci' in best_emb.hparams["regime"]:
            spatial = best_emb(torch.cat([batch.cell_data, batch.x], axis=-1))
    else:
            spatial = best_emb(batch.x)       
    # truth information
    e_bidir = torch.cat([batch.layerless_true_edges,
                        torch.stack([batch.layerless_true_edges[1],
                                    batch.layerless_true_edges[0]], axis=1).T], axis=-1)
    if(torch.cuda.is_available()):
        spatial = spatial.cuda()  
    # clustering = build_edges
    e_spatial = best_emb.clustering(spatial, best_emb.hparams["r_val"], best_emb.hparams["knn"])
    # label edges as true and false
    e_spatialn, y_cluster = graph_intersection(e_spatial, e_bidir)
    espt = e_spatialn.cpu().detach().numpy().T
    pid_np = batch.pid.detach().numpy()
    return pid_np, espt, y_cluster
    
    
def emb_purity(best_emb,batch):
    
    if 'ci' in best_emb.hparams["regime"]:
            spatial = best_emb(torch.cat([batch.cell_data, batch.x], axis=-1))
    else:
            spatial = best_emb(batch.x)
    
    # truth information
    e_bidir = torch.cat([batch.layerless_true_edges,
                        torch.stack([batch.layerless_true_edges[1],
                                    batch.layerless_true_edges[0]], axis=1).T], axis=-1)
    if(torch.cuda.is_available()):
        spatial = spatial.cuda()
        
    # clustering = build_edges
    e_spatial = best_emb.clustering(spatial, best_emb.hparams["r_val"], best_emb.hparams["knn"])
    
    # label edges as true and false
    e_spatial, y_cluster = graph_intersection(e_spatial, e_bidir)
    
    # add all truth edges "weight" times, which is 4,
    # to balance the number of truth and fake edges in one batch
    if(torch.cuda.is_available()):
        e_spatial = e_spatial.cuda()
        e_bidir = e_bidir.cuda()
    
    e_spatial = torch.cat([e_spatial,
        e_bidir.transpose(0,1).repeat(1,best_emb.hparams["weight"]).view(-1, 2).transpose(0,1)], axis=-1)
    y_cluster = np.concatenate([y_cluster.astype(int), np.ones(e_bidir.shape[1]*best_emb.hparams["weight"])])

    # extract emedding features of seed hits and neighbor hits
    seed_hits = spatial.index_select(0, e_spatial[1])
    neighbors = spatial.index_select(0, e_spatial[0])
    
    # extract true positives
    cluster_true = 2*len(batch.layerless_true_edges[0])
    cluster_true_positive = y_cluster.sum()
    cluster_positive = len(e_spatial[0])

    # calculate purity as true hits inside embedding radius / total hits inside embedding radius
    purity = cluster_true_positive/cluster_positive
    
    return purity

#############################################
#               ADD NOISE                   #
#############################################

def add_perc_noise(hits, truth, perc = 0.0):
    print(f"adding {perc}% noise")
    if perc >= 1.0:
        return hits,truth
    
    unique_ids = truth.particle_id.unique()
    track_ids_to_keep = unique_ids[np.where(unique_ids != 0)]
    noise_hits = unique_ids[np.where(unique_ids == 0)]
    where_to_keep = truth['particle_id'].isin(track_ids_to_keep)
    hits_reduced  = hits[where_to_keep]
    hit_ids_red = hits_reduced.hit_id.values
    noise_ids = hits[~where_to_keep].hit_id.values
    
    if perc <= 0.0:
        noise_ids = []
    else:
        num_rows = int(perc * noise_ids.shape[0])
        noise_ids = np.random.permutation(noise_ids)[:num_rows]

    #add noise
    hits_ids_noise = np.concatenate([hit_ids_red, noise_ids])
    
    noise_hits = hits[hits['hit_id'].isin(hits_ids_noise)]
    noise_truth = truth[truth['hit_id'].isin(hits_ids_noise)]
    #noise_cells = cells[cells['hit_id'].isin(noise_truth.hit_id.values)]
    
    return noise_hits, noise_truth

def get_noise(hits, truth):
    unique_ids = truth.particle_id.unique()
    track_ids_to_keep = unique_ids[np.where(unique_ids != 0)]
    where_to_keep = truth['particle_id'].isin(track_ids_to_keep)
    not_noise  = hits[where_to_keep]
    noise = hits[~where_to_keep]
    return noise, not_noise