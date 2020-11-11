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
    
def test_
    
    
    


    
    
    
    
    