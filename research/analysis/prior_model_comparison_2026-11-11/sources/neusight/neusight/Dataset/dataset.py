import pandas as pd
import numpy as np
import re
import os
import fcntl
import torch
from torch.utils.data import Dataset as TorchDataset
from pathlib import Path

ops_dict = {
    "add" : 1.,
    "addu": 1.,
    "mul" : 1.,
    "mulu": 1.,
    "pow" : 1.,
    "powu": 1.,
    "div" : 1.,
    "divu": 1.,
    "tanh": 1.,
    "ln"  : 6., # mean, var, sum, div, sqrt, acc
    "softmax" : 5.,
    "relu" : 1.,
    "fused_relu_add" : 2.,
    "gelu" : 1.,
    "MEM" : 0.,
}

inter_dict = {
    "ln"  : 0., # one pass for mean/var
    "softmax" : 0., # one pass for max
    "add" : 0.,
    "addu": 0.,
    "mul" : 0.,
    "mulu": 0.,
    "pow" : 0.,
    "powu": 0.,
    "div" : 0.,
    "divu": 0.,
    "tanh": 0.,
    "relu" : 0.,
    "fused_relu_add" : 0.,
    "gelu" : 0.,
    "MEM" : 0.,
}
        
def read_tile(opname, kname, B, H, GridX, GridY, GridZ):
    kname = kname.lower()

    if opname == "bmm" or opname == "linear":
        # amd
        p = re.compile('_mt[0-9]+x[0-9]+') 
        match = p.search(kname)
        if match:
            res = match.group()
            res = res[3:]
            res = res.split("x")
            res = list(map(int, res))
            return pd.Series(res)
        
        # sgemm cutlass
        p = re.compile('sgemm_(\d+)+x(\d+)')
        match = p.search(kname)
        if match:
            res = match.group().split("_")[1].split("x")
            res = list(map(int, res))
            return pd.Series(res)

        # try h100 style kernel
        p = re.compile('tilesize(\d+)+x(\d+)')
        match = p.search(kname)
        if match:
            res = match.group()[len("tilesize"):]
            res = res.split("x")[:2]
            res = list(map(int, res))
            return pd.Series(res)

        # it should not reach here
        print(kname)
        assert(0)
    elif opname == "ln" or opname == "softmax":
        return pd.Series([H, 0])
    else:
        tile = np.ceil(B * H / (GridX * GridY * GridZ))
        tile = 2 ** (np.ceil(np.log2(tile)))
        return pd.Series([tile, 0])

def multiplyList(myList):
    # Multiply elements one by one
    result = 1
    for x in myList:
        result = result * x
    return result

def count_ops_mem(opname, B, H):
    if opname in ["bmm", "linear"]:
        return 0, 0

    unary_vec_ops  = ["ln", "softmax", "tanh", "addu", "mulu", "powu", "divu", "relu", "gelu"]

    num_input_elem = B*H
    if opname not in unary_vec_ops:
        num_input_elem *= 2
    num_output_elem = B*H
    num_inter_elem = B*H*inter_dict[opname]

    memPerO = (num_input_elem + num_output_elem + num_inter_elem) * 4 / num_output_elem
    opsPerO = ops_dict[opname]

    return opsPerO, memPerO

class Dataset(TorchDataset):
    def __init__(self, dataset_path, device_list=None):
        dataset_path =  Path(dataset_path)

        # cache directory for
        cache_path = Path.home()/".cache"/"neusight"/(str(dataset_path.resolve()).replace("/", "%"))
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        flock = open(str(cache_path)+".lock", "a")
        fcntl.flock(flock, fcntl.LOCK_EX)

        # look for device configurations
        import glob
        files = glob.glob(str(dataset_path.parent/"device_configs"/"*.json"), recursive=True)
        device_df = []
        for file in files:
            df = pd.read_json(file, typ='series', orient="columns")
            device_df.append(df)
        device_df = pd.concat(device_df, axis=1)
        device_df = device_df.transpose()
        self.device_df = device_df
        # print("!!!")
        # print("device df : ", self.device_df)
        # print("!!!")

        # only one process needs to do this
        if not self.check_exists(cache_path) or self.check_stale(dataset_path, cache_path):
            df = pd.read_csv(dataset_path)
            df = self.prepare(df, device_list)
            with open(cache_path, "w") as f:
                df.to_csv(f, index=False)
            print(f"created {f}")

        self.df = pd.read_csv(cache_path)
        with open(str(cache_path)+".mean", "w") as f:
            self.df.mean(numeric_only=True).to_csv(f)
            self.df.std(numeric_only=True).to_csv(f)

        fcntl.flock(flock, fcntl.LOCK_UN)
        
        assert(self.df is not None)

        y = self.df["Latency"].values
        self.y = np.array(y)

        o = self.df["OPName"].values
        self.o = o

        # set metadata
        m1 = self.df["Tile1"].values
        m1 = np.array(m1)
        m2 = self.df["Tile2"].values
        m2 = np.array(m2)
        self.m = np.stack((m1, m2), axis=1)

    def prepare(self, df, device_list):
        if "Kernels" in df.columns:
            df = df.drop("Kernels", axis=1)

        # extract device
        if device_list is not None:
            df = df[df["Device"].isin(device_list)]

        # append device features
        df = df.merge(self.device_df, how='left', left_on=['Device'], right_on=['Device']) 
        if (df.isnull().values.any()):
            assert(0)
            print("null entries")

        # ms to sec
        df["Latency"] = df["Latency"] / 1000

        # dummy h column
        if 'H' not in df.columns:
            df['H'] = 0
        df[["Tile1", "Tile2"]] = df.apply(lambda x: read_tile(x["OPName"], x["Kernel Name"], x["B"], x["H"], x["Grid x"], x["Grid y"], x["Grid z"]), axis=1, result_type="expand")

        # add dummy columns
        df["MemPerO"] = 0
        df["OpsPerO"] = 0

        df[["OpsPerO", "MemPerO",]] = df.apply(lambda x: count_ops_mem(x["OPName"], x["B"], x["H"],), axis=1, result_type="expand")

        return df
    
    def check_stale(self, dataset_path, cache_path):
        # check if cached one is stale
        if os.path.getctime(cache_path) < os.path.getctime(dataset_path):
            return None
    
    def check_exists(self, path):
        return os.path.isfile(path)

    def load_cached(self, dataset_path, cache_path):
        # check if cached one exists
        if not self.check_exists(cache_path):
            return None
        
        # check if cached one is stale
        if self.check_stale(dataset_path, cache_path):
            return None
        
        # load cached one
        return pd.read_csv(cache_path)

    def get_df(self):
        return self.df
    
    def set_features(self, features):
        self.features = features
        x = self.df[features].values
        self.x = np.array(x)
    
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        o = self.o[idx]
        x = torch.from_numpy(np.array(self.x[idx]).astype(np.float32),)
        m = torch.from_numpy(np.array(self.m[idx]).astype(np.float32),)
        y = float(self.y[idx])

        return o, x, m, y
