from pytorch_lightning import Trainer, seed_everything
import pandas as pd
import pd.DataFrame as df
import numpy as np
import random as rd

# generate random data
seed_everything(42)

def data_generator(data_path, event_name, nhits):
    print(f"creating random event {event_name} of {nhits} hits in {data_path}...")
    
    # create truth.csv
    
    
    # create cells.csv
    
    # create hits.csv
    
    # create partciles.csv
    
    
def create_truth(nhits):
    truth_data = []
    truth_columns = ['hit_id', 'particle_id', 'tx', 'ty', 'tz', 'tpx', 'tpy', 'tpz', 'weight']
    # create hitids
    truth_data.append(np.arange(1,nhits,1))
    # create pids
    random.sample(range(10000000,99999999), nhits)
    
    
    truth_table = df(data=truth_data, columns=truth_columns)