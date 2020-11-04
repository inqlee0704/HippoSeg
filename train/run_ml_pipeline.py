"""
This file contains code that will kick off training and testing processes
"""
import os
import json

from experiments.UNetExperiment import UNetExperiment
from data_prep.HippocampusDatasetLoader import LoadHippocampusData

class Config:
    """
    Holds configuration parameters
    """
    def __init__(self):
        self.name = "Basic_unet"
        # ROOT PATH: 
        self.root_dir = r"../data/TrainProc"
        self.n_epochs = 10
        self.learning_rate = 0.0002
        self.batch_size = 8
        self.patch_size = 64
        self.test_results_dir = "RESULTS"

if __name__ == "__main__":
    # Get configuration

    c = Config()

    # Load data
    print("Loading data...")

    data = LoadHippocampusData(c.root_dir, y_shape = c.patch_size, z_shape = c.patch_size)

    keys = range(len(data))
    # Data Split: | 60% Training | 20% Validation | 20% Test | 
    split = dict()

    s0 = 0
    s1 = int(0.6*len(data))
    s2 = int(0.8*len(data))
    s3 = len(data)
    split['train'] = range(s0,s1)
    split['val'] = range(s1,s2)
    split['test'] = range(s2,s3)    
    # print(data[split["train"]][0]['image'][0])
    
    # Set up and run experiment
    exp = UNetExperiment(c, split, data)

    # run training
    exp.run()

    # prep and run testing
    results_json = exp.run_test()

    results_json["config"] = vars(c)

    with open(os.path.join(exp.out_dir, "results.json"), 'w') as out_file:
        json.dump(results_json, out_file, indent=2, separators=(',', ': '))

