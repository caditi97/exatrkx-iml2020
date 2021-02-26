import sys
import os
import torch
import pytorch_lightning as pl
import yaml
import importlib
import trackml
import trackml.dataset
import numpy as np

# from torch_cluster import radius_graph
# from utils_torch import build_edges
# from embedding.layerless_embedding import *
# from embedding.embedding_base import *
# # from src.filter import filter_base
# # from src.filter.vanilla_filter import *

# system import
import pkg_resources
import yaml
import pprint
import random
random.seed(1234)
import pandas as pd
import itertools
import matplotlib.pyplot as plt
# %matplotlib widget

# 3rd party
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from trackml.dataset import load_event
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint


# local import
# from heptrkx.dataset import event as master
from exatrkx import config_dict # for accessing predefined configuration files
from exatrkx import outdir_dict # for accessing predefined output directories
from exatrkx.src import utils_dir

# for preprocessing
from exatrkx import FeatureStore
from exatrkx.src import utils_torch

# for embedding
from exatrkx import LayerlessEmbedding
from exatrkx.src import utils_torch
from utils_torch import *

# for filtering
from exatrkx import VanillaFilter

# for GNN
import tensorflow as tf
from graph_nets import utils_tf
from exatrkx import SegmentClassifier
import sonnet as snt

# for labeling
from exatrkx.scripts.tracks_from_gnn import prepare as prepare_labeling
from exatrkx.scripts.tracks_from_gnn import clustering as dbscan_clustering

# track efficiency
from trackml.score import _analyze_tracks
from exatrkx.scripts.eval_reco_trkx import make_cmp_plot, pt_configs, eta_configs
from functools import partial

import matplotlib.pyplot as plt

#default values
os.environ['TRKXINPUTDIR']="/global/cfs/cdirs/m3443/data/trackml-kaggle/train_10evts"
os.environ['TRKXOUTPUTDIR']= "/global/cfs/projectdirs/m3443/usr/caditi97/iml2020/outtest"


#############################################
#                  UTILS                    #
#############################################

def preprocess():
    action = 'build'

    config_file = pkg_resources.resource_filename(
                        "exatrkx",
                        os.path.join('configs', config_dict[action]))
    with open(config_file) as f:
        b_config = yaml.load(f, Loader=yaml.FullLoader)

    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(b_config)
    
    b_config['pt_min'] = 0
    b_config['endcaps'] = True
    b_config['n_workers'] = 1
    b_config['n_files'] = 1
    
    preprocess_dm = FeatureStore(b_config)
    preprocess_dm.prepare_data()

def emb_eval(embed_ckpt_dir,data):
    e_ckpt = torch.load(embed_ckpt_dir, map_location='cpu')
    e_config = e_ckpt['hyper_parameters']
    pp = pprint.PrettyPrinter(indent=4)
    #pp.pprint(e_config)
    
    e_config = e_ckpt['hyper_parameters']
    e_config['clustering'] = 'build_edges'
    e_config['knn_val'] = 500
    e_config['r_val'] = 1.7
    
    e_model = LayerlessEmbedding(e_config)
    e_model.load_state_dict(e_ckpt["state_dict"])
    
    e_model.eval()
    
    spatial = e_model(torch.cat([data.cell_data, data.x], axis=-1))
    
    if(torch.cuda.is_available()):
        spatial = spatial.cuda()

    e_spatial = utils_torch.build_edges(spatial, e_model.hparams['r_val'], e_model.hparams['knn_val'])
    
    e_spatial = e_spatial.cpu().numpy()
    
    R_dist = torch.sqrt(data.x[:,0]**2 + data.x[:,2]**2) # distance away from origin...
    e_spatial = e_spatial[:, (R_dist[e_spatial[0]] <= R_dist[e_spatial[1]])]
    
    return e_spatial

def filtering(filter_ckpt_dir,data,e_spatial):
    f_ckpt = torch.load(filter_ckpt_dir, map_location='cpu')
    f_config = f_ckpt['hyper_parameters']
    pp = pprint.PrettyPrinter(indent=4)
    #pp.pprint(f_config)
    
    f_config['train_split'] = [0, 0, 1]
    f_config['filter_cut'] = 0.18
    
    f_model = VanillaFilter(f_config)
    f_model.load_state_dict(f_ckpt['state_dict'])
    
    f_model.eval()
    
    emb = None # embedding information was not used in the filtering stage.
    output = f_model(torch.cat([data.cell_data, data.x], axis=-1), e_spatial, emb).squeeze()
    output = torch.sigmoid(output)
    
    return output, f_model

def build_graph(output, f_model,data, e_spatial, gnn_ckpt_dir,ckpt_idx,dbscan_epsilon, dbscan_minsamples):
    edge_list = e_spatial[:, output > f_model.hparams['filter_cut']]
    
    n_nodes = data.x.shape[0]
    n_edges = edge_list.shape[1]
    nodes = data.x.numpy().astype(np.float32)
    edges = np.zeros((n_edges, 1), dtype=np.float32)
    senders = edge_list[0]
    receivers = edge_list[1]
    
    input_datadict = {
    "n_node": n_nodes,
    "n_edge": n_edges,
    "nodes": nodes,
    "edges": edges,
    "senders": senders,
    "receivers": receivers,
    "globals": np.array([n_nodes], dtype=np.float32)
    }
    
    input_graph = utils_tf.data_dicts_to_graphs_tuple([input_datadict])
    
    print(f' APPLYING GNN.....')
    
    num_processing_steps_tr = 8
    optimizer = snt.optimizers.Adam(0.001)
    model = SegmentClassifier()

    output_dir = gnn_ckpt_dir
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    ckpt_manager = tf.train.CheckpointManager(checkpoint, directory=output_dir, max_to_keep=10)
    status = checkpoint.restore(ckpt_manager.checkpoints[ckpt_idx]).expect_partial()
    print("Loaded {} checkpoint from {}".format(ckpt_idx, output_dir))
    
    outputs_gnn = model(input_graph, num_processing_steps_tr)
    output_graph = outputs_gnn[-1]
    
    print(f'TRACK LABELLING.....')
    
    input_matrix = prepare_labeling(tf.squeeze(output_graph.edges).numpy(), senders, receivers, n_nodes)
    predict_tracks = dbscan_clustering(data.hid, input_matrix, dbscan_epsilon, dbscan_minsamples)
    
    return predict_tracks

def track_eff(evt_path, predict_tracks,min_hits,frac_reco_matched, frac_truth_matched,evtid=0):
    hits, particles, truth = load_event(evt_path, parts=['hits', 'particles', 'truth'])
    hits = hits.merge(truth, on='hit_id', how='left')
    hits = hits[hits.particle_id > 0] # remove noise hits
    hits = hits.merge(particles, on='particle_id', how='left')
    hits = hits[hits.nhits >= min_hits]
    particles = particles[particles.nhits >= min_hits]
    par_pt = np.sqrt(particles.px**2 + particles.py**2)
    momentum = np.sqrt(particles.px**2 + particles.py**2 + particles.pz**2)
    ptheta = np.arccos(particles.pz/momentum)
    peta = -np.log(np.tan(0.5*ptheta))
    
    tracks = _analyze_tracks(hits, predict_tracks)
    
    purity_rec = np.true_divide(tracks['major_nhits'], tracks['nhits'])
    purity_maj = np.true_divide(tracks['major_nhits'], tracks['major_particle_nhits'])
    good_track = (frac_reco_matched < purity_rec) & (frac_truth_matched < purity_maj)

    matched_pids = tracks[good_track].major_particle_id.values
    score = tracks['major_weight'][good_track].sum()

    n_recotable_trkx = particles.shape[0]
    n_reco_trkx = tracks.shape[0]
    n_good_recos = np.sum(good_track)
    matched_idx = particles.particle_id.isin(matched_pids).values
    
#     print("----------")
#     print("Processed {} events from {}".format(evtid, utils_dir.inputdir))
#     print("Reconstructable tracks:         {}".format(n_recotable_trkx))
#     print("Reconstructed tracks:           {}".format(n_reco_trkx))
#     print("Reconstructable tracks Matched: {}".format(n_good_recos))
#     print("Tracking efficiency:            {:.4f}".format(n_good_recos/n_recotable_trkx))
#     print("Tracking purity:               {:.4f}".format(n_good_recos/n_reco_trkx))
#     print("----------")
    
    return matched_idx, peta, par_pt

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
    
def plot_matched(xarray, yarray, xlegend, ylegend, configs, xlabel, ylabel, ratio_label, outname, ax1,ax2,c):
    m_vals, bins, _ = ax.hist(xarray, **configs, label=xlegend)
    n_vals, _, _ = ax.hist(yarray, **configs, label=ylegend)
    ax1.set_xlabel(xlabel, fontsize=fontsize)
    ax1.set_ylabel(ylabel, fontsize=fontsize)
    plt.legend()
    plt.savefig("{}.pdf".format(outname))
    
    ratio, ratio_err = get_ratio(m_vals, n_vals)
    xvals = [0.5*(x[1]+x[0]) for x in pairwise(bins)][1:]
    xerrs = [0.5*(x[1]-x[0]) for x in pairwise(bins)][1:]
    # print(xvals)
    ax2.errorbar(xvals, ratio, yerr=ratio_err, fmt='o', xerr=xerrs, lw=2,c=c)
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(ratio_label)
    ax2.set_yticks(np.arange(0.5, 1.05, step=0.05))
    ax2.set_ylim(0.5, 1.05)
    # ax.text(1, 0.8, "bins: [{}] GeV".format(", ".join(["{:.1f}".format(x) for x in pt_bins[1:]])))
    plt.grid(True)
    plt.savefig("{}_ratio.pdf".format(outname))
    
    
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

def get_emb_metrics(data_path, emb_model,r_val=1.7,knn_val=500):
    data = torch.load(data_path)
    spatial = emb_model(torch.cat([data.cell_data, data.x], axis=-1))
    
    if(torch.cuda.is_available()):
        spatial = spatial.cuda()
        
    e_spatial = utils_torch.build_edges(spatial, r_val, knn_val)
    e_spatial_np = e_spatial.cpu().numpy()
    
    # remove R dist from out to in
    R_dist = torch.sqrt(data.x[:,0]**2 + data.x[:,2]**2)
    
    e_spatial_np = e_spatial_np[:, (R_dist[e_spatial_np[0]] <= R_dist[e_spatial_np[1]])]
    e_bidir = torch.cat([data.layerless_true_edges,torch.stack([data.layerless_true_edges[1],
                        data.layerless_true_edges[0]], axis=1).T], axis=-1)
    e_spatial_n, y_cluster = graph_intersection(torch.from_numpy(e_spatial_np), e_bidir)
    
    cluster_true = len(data.layerless_true_edges[0])
    cluster_true_positive = y_cluster.sum()
    cluster_positive = len(e_spatial_n[0])
    purity = cluster_true_positive/cluster_positive
    eff = cluster_true_positive/cluster_true
    
    print("-----------")
    print(f"cluster true = {cluster_true}")
    print(f"cluste true positive = {cluster_true_positive}")
    print(f"cluster positive = {cluster_positive}")
    print(f"purity = {purity}")
    print(f"efficiency = {eff}")
    
    return purity, eff

def get_lvl_emb(mypath,emb_model,r_val=1.7,knn_val=500):
    events = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    p_lvl = []
    e_lvl = []
    for evt in events:
        data_path = join(mypath,evt)
        p, e = get_emb_metrics(data_path, emb_model,r_val,knn_val) 
        p_lvl.append(p)
        e_lvl.append(e)
    
    return np.mean(p_lvl), np.mean(e_lvl)

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