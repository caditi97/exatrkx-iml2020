# System imports
import sys
import os

# 3rd party imports
import torch
import torch.nn as nn
from torch.nn import Linear
from torch_cluster import radius_graph

from torch_geometric.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback


# Local imports
from exatrkx.src.filter.filter_base import FilterBaseBalanced
from exatrkx.src.utils_torch import graph_intersection

class VanillaFilter(FilterBaseBalanced):

    def __init__(self, hparams):
        super().__init__(hparams)
        '''
        Initialise the Lightning Module that can scan over different filter training regimes
        '''

        # Construct the MLP architecture
        self.input_layer = Linear(hparams["in_channels"]*2 + hparams["emb_channels"]*2, hparams["hidden"])
        layers = [Linear(hparams["hidden"], hparams["hidden"]) for _ in range(hparams["nb_layer"]-1)]
        self.layers = nn.ModuleList(layers)
        self.output_layer = nn.Linear(hparams["hidden"], 1)
        self.layernorm = nn.LayerNorm(hparams["hidden"])
        self.batchnorm = nn.BatchNorm1d(num_features=hparams["hidden"], track_running_stats=False)
        self.act = nn.Tanh()

    def forward(self, x, e, emb=None):
        if emb is not None:
            x = self.input_layer(torch.cat([x[e[0]], emb[e[0]], x[e[1]], emb[e[1]]], dim=-1))
        else:
            x = self.input_layer(torch.cat([x[e[0]], x[e[1]]], dim=-1))
        for l in self.layers:
            x = l(x)
            x = self.act(x)
            if self.hparams["layernorm"]: x = self.layernorm(x) #Option of LayerNorm
            if self.hparams["batchnorm"]: x = self.batchnorm(x) #Option of Batch
        x = self.output_layer(x)
        return x

class FilterInferenceCallback(Callback):
    def __init__(self):
        self.output_dir = None
        self.overwrite = False

    def on_train_start(self, trainer, pl_module):
        # Prep the directory to produce inference data to
        self.output_dir = pl_module.hparams.output_dir
        self.datatypes = ["train", "val", "test"]
        os.makedirs(self.output_dir, exist_ok=True)
        [os.makedirs(os.path.join(self.output_dir, datatype), exist_ok=True) for datatype in self.datatypes]

    def on_train_end(self, trainer, pl_module):
        print("Training finished, running inference to filter graphs...")

        # By default, the set of examples propagated through the pipeline will be train+val+test set
        datasets = {"train": pl_module.trainset, "val": pl_module.valset, "test": pl_module.testset}
        total_length = sum([len(dataset) for dataset in datasets.values()])
        batch_incr = 0

        pl_module.eval()
        with torch.no_grad():
            for set_idx, (datatype, dataset) in enumerate(datasets.items()):
                for batch_idx, batch in enumerate(dataset):
                    percent = (batch_incr / total_length) * 100
                    sys.stdout.flush()
                    sys.stdout.write(f'{percent:.01f}% inference complete \r')
                    # print(batch)

                    # print(not os.path.exists(os.path.join(self.output_dir, datatype, batch.event_file[-4:])))
                    # print(self.overwrite)
                    #
                    # print(os.path.join(self.output_dir, datatype, batch.event_file[-4:]))
                    if (not os.path.exists(os.path.join(self.output_dir, datatype, batch.event_file[-4:]))) or self.overwrite:
                        batch = batch.to(pl_module.device) #Is this step necessary??
                        batch = self.construct_downstream(batch, pl_module)
                        self.save_downstream(batch, pl_module, datatype)
                        del batch
                        torch.cuda.empty_cache()

                    batch_incr += 1

    def construct_downstream(self, batch, pl_module):

        emb = (None if (pl_module.hparams["emb_channels"] == 0)
               else batch.embedding)  # Does this work??

        output = pl_module(torch.cat([batch.cell_data, batch.x], axis=-1), batch.e_radius, emb).squeeze() if ('ci' in pl_module.hparams["regime"]) else pl_module(batch.x, batch.e_radius, emb).squeeze()
        y_pid = batch.pid[batch.e_radius[0]] == batch.pid[batch.e_radius[1]]

        cut_indices = output > pl_module.hparams["filter_cut"]
        batch.e_radius = batch.e_radius[:, cut_indices]
        batch.y_pid = y_pid[cut_indices]
        batch.y = batch.y[cut_indices]
        return batch

    def save_downstream(self, batch, pl_module, datatype):

        with open(os.path.join(self.output_dir, datatype, batch.event_file[-4:]), 'wb') as pickle_file:
            torch.save(batch, pickle_file)