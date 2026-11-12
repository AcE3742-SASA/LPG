import torch
from .mlp_wave import MLPWave

class MLPWaveVec(MLPWave):
    def __init__(self, config, tag, device):
        num_tile_inputs = 1
        super().__init__(config,tag=tag,device=device,num_tile_inputs=num_tile_inputs) # name and feature

    def compute_ops(self, x, opname, h):
        opsPerO = self.get_feature(x, "OpsPerO")
        op_flops = opsPerO * h / (10**9) # GFLOPs
        return op_flops

    def compute_mem(self, x, opname, h):
        memPerO = self.get_feature(x, "MemPerO")
        op_mem = memPerO * h / (2**30) # in GB
        return op_mem

    def comptue_op_arithinten(self, x, opname, tiles):
        # op arithinten
        h = self.get_feature(x, "H")
        op_flops = self.compute_ops(x=x, opname=opname, h=self.ceil_multiple(h, tiles))
        op_mem = self.compute_mem(x=x, opname=opname, h=h)
        op_arithinten  = op_flops / op_mem
    
        return op_arithinten

    def compute_ops_per_wave(self, x, tiles, opname):
        num_sm = self.get_feature(x, "Num_Sm")

        ops_per_wave = num_sm * self.compute_ops(x=x, opname=opname, h=tiles)

        return ops_per_wave

    def compute_num_block(self, x, tile):
        # input dims
        b = self.get_feature(x, "B")
        h = self.get_feature(x, "H")
        
        num_block = b * torch.ceil(h / tile)

        return num_block

    def get_tiles(self, x, tiles, culib, opname):
        # tiles
        if tiles is None:
            # assert(culib is not None)
            assert(len(opname) == 1) # single batch for now
            if opname[0] in ["ln", "softmax"]:
                tiles = self.get_feature(x, "H")
            else:
                tiles = self.meta_table.get_tile(x, culib=culib, opname=opname)
                tiles = tiles[0]
                tiles = tiles.to(self.device)
        else:
            tiles = tiles[:,0] # take the firstone
            # print(tiles)
        tiles = tiles.view(-1,1)
        
        self.last_tiles = tiles
        
        return tiles

    def compute_tile_ops(self, x, opname, tiles):
        tile_flops = self.compute_ops(x=x, opname=opname, h=tiles) # in GFLOPs
        return tile_flops

    def compute_tile_mem(self, x, opname, tiles):
        tile_mem = self.compute_mem(x=x, opname=opname, h=tiles) * 1024 # to MB
        return tile_mem
