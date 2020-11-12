import unittest

import os
import yaml
# import pprint

from pytorch_lightning import Trainer
import trackml
from trackml.dataset import load_event
# from exatrkx import config_dict
# from exatrkx import outdir_dict
# from exatrkx.scripts.run_lightning as run_l
# from run_l import build

# local imports 
from exatrkx.src.utils_robust import emb_purity
from utils_robust import *

og_event = '/global/cfs/cdirs/m3443/data/trackml-kaggle/train_10evts/'
event_name = 'event000001000'
event_num = '1000'
noise_dir = '/global/cfs/projectdirs/m3443/usr/caditi97/iml2020/'
emb_ckpt_path = '/global/cfs/cdirs/m3443/data/lightning_models/embedding/checkpoints/epoch=10.ckpt'

def test_noise_reduced(perc):
    in_hits, in_cells, in_particles, in_truth = load_event(og_event + event_name)
    exp_hits, exp_truth = add_perc_noise(in_hits,in_truth,perc)
    assert len(exp_hits) <= len(in_hits), "reduced hits %r are not less than original %r" % (len(exp_hits),len(in_hits))
    
def test_noise_perc(perc):
    in_hits, in_cells, in_particles, in_truth = load_event(og_event + event_name)
    in_noise, in_not_noise = get_noise(in_hits,in_truth)
    num_rows = int(perc * in_noise.shape[0])
    
    out_hits, out_truth = add_perc_noise(in_hits,in_truth,perc)
    out_noise, out_not_noise = get_noise(out_hits,out_truth)
    assert num_rows == len(out_noise), "added noise %r is not the correct percentage %r" % (len(out_noise), perc)
    
def test_cluster_noise(perc):
    
    emb_ckpt = get_emb_ckpt(emb_ckpt_path, [8,1,1], 'build_edges')
    best_emb = load_cktp(emb_ckpt, emb_ckpt_path, True)
    
    d_path = f"/global/cfs/projectdirs/m3443/usr/caditi97/iml2020/out{perc}/feature_store/{event_num}"
    batch = torch.load(d_path)
    
    pid_np, espt, y_cluster = get_cluster(best_emb,batch)
    
    exp = 0
    for idx in range(len(y_cluster)):
        noise1 = 0
        noise2 = 0
        if y_cluster[idx]:
            hitid1 = espt[idx][0]
            hitid2 = espt[idx][1]
            if pid_np[hitid1] == 0:
                noise1+=1
            if pid_np[hitid2] == 0:
                noise2+=1
    
    assert (noise1==exp and noise2==exp), "cluster selected noise hits at %r with hit1 %r and hit2 %r in dir %r" %(perc, noise1, noise2, d_path)
    
    
    
    
    


    
    
    
    
    