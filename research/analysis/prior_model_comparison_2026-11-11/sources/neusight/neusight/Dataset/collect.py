import torch
from torch.profiler import profile
import json
import time
from pathlib import Path
import os
import pandas as pd
import random
import uuid
import glob
import re
import shutil

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = False

starter = torch.cuda.Event(enable_timing=True)
ender = torch.cuda.Event(enable_timing=True)
active = 25
sample = 5
assert(active > sample)

conv_ops = ["conv"]
mm_ops = ["bmm", "linear"]
binary_vec_ops = ["add", "mul", "pow", "div",]
unary_vec_ops  = ["ln", "softmax", "tanh", "addu", "mulu", "powu", "divu", "relu", "gelu",]
vec_ops = binary_vec_ops + unary_vec_ops

elem_ops = ["add", "mul", "pow", "div",] + ["addu", "mulu", "powu", "divu", "relu", "gelu", "tanh",]

kname = {
    "bmm" : (re.compile('(a|b|c)(i|j|k|l)(i|j|k|l)(i|j|k|l)_(a|b|c)(i|j|k|l)(i|j|k|l)(i|j|k|l)_(a|b|c)(i|j|k|l)(i|j|k|l)(i|j|k|l).*'), re.compile(r'.*sgemm_(\d+)+x(\d+).*'), re.compile(r'.*tilesize(\d+)+x(\d+).*')),
    "linear" : (re.compile('(a|b|c)(i|j|k|l)(i|j|k|l)(i|j|k|l)_(a|b|c)(i|j|k|l)(i|j|k|l)(i|j|k|l)_(a|b|c)(i|j|k|l)(i|j|k|l)(i|j|k|l).*'), re.compile(r'.*sgemm_(\d+)+x(\d+).*'), re.compile(r'.*tilesize(\d+)+x(\d+).*')),
    "add" : (re.compile('.*elementwise.*'), re.compile('.*add.*')),
    "addu" : (re.compile('.*elementwise.*'), re.compile('.*add.*')),
    "mul" : (re.compile('.*elementwise.*'), re.compile('.*mul.*')),
    "mulu" : (re.compile('.*elementwise.*'), re.compile('.*mul.*')),
    "pow"  : (re.compile('.*elementwise.*'), re.compile('.*pow.*')),
    "powu" : (re.compile('.*elementwise.*'), re.compile('.*pow.*')),
    "div" : (re.compile('.*elementwise.*'), re.compile('.*div.*')),
    "divu" : (re.compile('.*elementwise.*'), re.compile('.*div.*')),
    "tanh" : (re.compile('.*elementwise.*'), re.compile('.*tanh.*')),
    "ln" : (re.compile('.*layer_norm.* | .*layernorm.*'), ),
    "softmax" :(re.compile('.*softmax.*'), ),
    "dropout" : (re.compile('.*dropout.*'), ),
    "relu" : (re.compile('.*clamp.*'), ),
    "gelu" : (re.compile('.*elementwise.*'), re.compile('.*gelu.*')),
    # "conv" : (re.compile('.*winograd_(\d+)+x(\d+).*'), re.compile('.*scudnn_(\d+)+x(\d+).*'), 
    #             re.compile('.*convolve_sgemm.*'), re.compile('.*sgemm_(\d+)+x(\d+).*'),
    #             re.compile('.*gcgemm_(\d+)+x(\d+).*'), re.compile('.*implicit_gemm.*'),
    #             re.compile('.*tilesize(\d+)+x(\d+)x(\d+)+.*'),
    #             ),
}


def measure_once(op, input_batch):
    starter.record()
    out = op(*input_batch)
    ender.record()
    torch.cuda.synchronize()
    return starter.elapsed_time(ender)

def profile_kernel(opname, dim):
    
    b, h, m, n, k = 1, 1, 1, 1, 1 # dummy
    i_c, o_c, k_s, i_s, stride, padding = 1, 1, 1, 1, 1, 1 # dummy

    if opname == "bmm":
        b, m, n, k = dim
        input_batch = (torch.randn(b,m,n,device="cuda"), torch.randn(b,n,k,device="cuda"))
    elif opname == "linear":
        b, m, n, k = dim
        input_batch = (torch.randn(b,m,n,device="cuda"),)
    elif opname in vec_ops:
        b, h = dim
        if opname in binary_vec_ops:
            input_batch = (torch.zeros(b,h,device="cuda"), torch.zeros(b,h,device="cuda"))
        elif opname in unary_vec_ops:
            input_batch = (torch.zeros(b,h,device="cuda"), )
    elif opname == "conv":
        b, i_c, o_c, k_s, i_s, stride, padding = dim
        input_batch = (torch.zeros(b,i_c,i_s,i_s,device="cuda"),)
    else:
        assert(0)

    randn = 1 - random.random()
    op = {
        "conv" : torch.nn.Conv2d(i_c, o_c, k_s, stride=stride, padding=padding, bias=False, device="cuda"),
        "bmm" : torch.bmm,
        "linear" : torch.nn.Linear(n,k,bias=True,device="cuda"),
        # "linear" : torch.nn.Linear(n,k,bias=False,device="cuda"),
        "add" : torch.add,
        "mul" : torch.mul,
        "pow"  : torch.pow,
        "div" : torch.div,
        "addu" : lambda x : torch.add(x, randn),
        "mulu" : lambda x : torch.mul(x, randn),
        "powu" : lambda x : torch.pow(x, randn),
        "divu" : lambda x : torch.div(x, randn),
        "tanh" : torch.tanh,
        "ln" : torch.nn.LayerNorm(input_batch[0].shape[-1],device="cuda"),
        "softmax" : torch.nn.Softmax(),
        "dropout" : torch.nn.Dropout(p=0.5),
        "relu" : torch.nn.ReLU(),
        "gelu" : torch.nn.GELU(),
    }[opname]

    # measure
    latency_list = []
    for _ in range(active):
        latency_list.append(measure_once(op, input_batch))
    latency_list.sort()
    latency_list = latency_list[:sample]
    latency = sum(latency_list)/len(latency_list)

    # profile
    with profile() as prof: #with_stack=True,profile_memory=True,on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/resnet18'),
        out = op(*input_batch)
        torch.cuda.synchronize()
    fname = uuid.uuid4().hex
    prof.export_chrome_trace(f"/tmp/{fname}.json")
    # shutil.copy(f"/tmp/{fname}.json", "/home1/seonho/NeuSight/dev/amd/scripts/out/out.json")

    with open(f"/tmp/{fname}.json") as f:
        trace = json.load(f)

    knames = ""
    for e in trace["traceEvents"]:
        if e.get("cat") == "cuda_runtime" and "LaunchKernel" in e.get("name"):
            knames = knames + ";" + e["args"]["kernel"].lower()

        # if e.get("cat") == "kernel":
        #     knames = knames + ";" + e["name"].lower()

    for e in trace["traceEvents"]:
        if e.get("cat") == "cuda_runtime" and "LaunchKernel" in e.get("name"):
            for pat in kname[opname]:
                # print(pat)
                # print(e["name"])
                # print(pat.match(e["name"].lower()))
                if pat.match(e["args"]["kernel"].lower()) is not None:
                    return latency, e, knames

    print(f"no match for {opname}")
    print(f"skip dim {dim}")
    # print(trace["traceEvents"])
    # print(e["name"])

    shutil.copy(f"/tmp/{fname}.json", f"/home1/seonho/NeuSight/dev/amd/scripts/out/notfound/{opname}_{dim}.json")

    # assert(0)
    return latency, None, knames

def collect(opname, dims_path, out_dir_base):
    print("collecting", opname, flush=True)
    start_time = time.time()

    # collect metadata
    # device_name = device_names[torch.cuda.get_device_name()]
    device_name = torch.cuda.get_device_name().strip()
    if device_name == "AMD Instinct MI250X / MI250":
        device_name = "AMD Instinct MI250"
    cudnn_version = torch.backends.cudnn.version()
    torch_version, cuda_version = torch.__version__.split("+")
    torch_version = "p" + torch_version

    # set out_dir
    out_dir = Path(out_dir_base) 
    out_dir = out_dir/opname/torch_version
    out_dir.mkdir(parents=True, exist_ok=True)

    # open dims
    dims = []
    if not os.path.isfile(dims_path):
        print(f"{dims_path} not found")
        return

    with open(dims_path) as f:
        for line in f.readlines():
            line = line.strip()
            line = line.split(";")
            line = list(map(int,line))
            dims.append(line)

    entries = []

    # start measuring
    for i, dim in enumerate(dims):
        if (i%100 == 0):
            print(f"processing : {i}th", flush=True)
        try:
            with torch.no_grad():
                lat, event, knames = profile_kernel(opname, dim)
                if event is None:
                    print(f"skip {i}")
                    continue

# Operator Name,B,M,N,K,Kernel Name,Latency,Warps per SM,Grid x,Grid y,Grid z,Block x,Block y,Block z,Torch Version,CUDNN Version,Device,TileX,TileY,CUDA Version,OOD
            entry = {
                "OPName" : opname,
                "Latency" : lat,
                "Device" : device_name,
                "Kernel Name" : event["args"]["kernel"],
                "Grid x" : event["args"]["grid"][0],
                "Grid y" : event["args"]["grid"][1],
                "Grid z" : event["args"]["grid"][2],
                "Block x" : event["args"]["block"][0],
                "Block y" : event["args"]["block"][1],
                "Block z" : event["args"]["block"][2],
                "Kernels" : knames,
            }
            if opname in mm_ops:
                entry["B"], entry["M"], entry["N"], entry["K"] = dim
            elif opname in conv_ops:
                entry["B"], entry["Input_Channel"], entry["Output_Channel"], \
                    entry["Kernel_Szie"], entry["Input_Size"], entry["Stride"], entry["Padding"] = dim
            else:
                entry["B"], entry["H"] = dim

            entries.append(entry)

        except Exception as e:
            if "out of memory" in str(e):
                # print("out of memory")
                pass # oom
            else:
                print(opname)
                print(dim)
                raise e

    # dump json
    df = pd.DataFrame(entries)
    print(device_name)
    df.to_csv(out_dir/f"{device_name}.csv", index=False)

    # total program time
    ptime = time.time() - start_time
    print(f"time taken : {ptime/60.}")

def dataset_merge(merge_dir, out_fname):
    files = glob.glob(str(merge_dir/"**"/"*.csv"), recursive=True)
    dfs = []
    for file in files:
        df = pd.read_csv(file)
        dfs.append(df)
    df_merged = pd.concat(dfs)

    # df_merged["Kernel Name"] =df_merged["Kernels"] # ?

    df_merged.to_csv(out_fname, index=False)
    return df_merged

def vec_merge(merge_dir, out_fname):
    dfs = []
    for op in vec_ops:
        file = Path(merge_dir)/(op+".csv")
        df = pd.read_csv(file)
        dfs.append(df)
    df_merged = pd.concat(dfs)
    df_merged.to_csv(out_fname, index=False)
    return df_merged

def elem_merge(merge_dir, out_fname):
    dfs = []
    for op in elem_ops:
        file = Path(merge_dir)/(op+".csv")
        df = pd.read_csv(file)
        dfs.append(df)
    df_merged = pd.concat(dfs)

    def is_large(b,h):
        return b > 8192 and h > 512
    df_merged["large"] = df_merged.apply(lambda x: is_large(x["B"], x["H"],), axis=1)
    df_merged = df_merged[df_merged["large"] == True]
    df_merged.to_csv(out_fname, index=False)

    return df_merged

def softmax_merge(merge_dir, out_fname):
    dfs = []
    for op in ["softmax"]:
        file = Path(merge_dir)/(op+".csv")
        df = pd.read_csv(file)
        dfs.append(df)
    df_merged = pd.concat(dfs)

    def is_large(b,h):
        return b > 4096 and h > 512
    df_merged["large"] = df_merged.apply(lambda x: is_large(x["B"], x["H"],), axis=1)
    df_merged = df_merged[df_merged["large"] == True]
    df_merged.to_csv(out_fname, index=False)

    return df_merged

def ln_merge(merge_dir, out_fname):
    dfs = []
    for op in ["ln"]:
        file = Path(merge_dir)/(op+".csv")
        df = pd.read_csv(file)
        dfs.append(df)
    df_merged = pd.concat(dfs)

    def is_large(b,h):
        return b > 4096 and h > 512
    df_merged["large"] = df_merged.apply(lambda x: is_large(x["B"], x["H"],), axis=1)
    df_merged = df_merged[df_merged["large"] == True]

    # df_merged = df_merged.query("OpName.str.contains(r'VEC(?!softmax|ln$)')").sum(axis=0)
    # df_merged = df_merged[df_merged["Kernel Name"] == "void at::native::(anonymous namespace)::vectorized_layer_norm_kernel<float, float>(int, float, float const*, float const*, float const*, float*, float*, float*)"]

    df_merged.to_csv(out_fname, index=False)

    return df_merged

def mark_ood(df):
    def is_ood(b,m,n,k):
        return b > 3072 or m > 3072 or n > 3072 or k > 3072
    df["OOD"] = df.apply(lambda x: is_ood(x["B"], x["M"], x["N"], x["K"]), axis=1)
    return df
