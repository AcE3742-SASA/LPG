import torch
import torch.nn as nn

class ModelBase(nn.Module):
    def __init__(self,config,tag,device):
        super().__init__()

        # read config
        self.config = config

        self.name = self.config["name"]
        if tag is not None:
            self.name = self.name + "-" + tag
        self.features = self.config["features"]
        self.device=device

    def forward(self, x):
        raise NotImplementedError

    def stats(self):
        return {}
    
    def set_record(self, x):
        return x
    
    def dump(self, dataset_name):
        pass

    def set_meta_table(self, table_path):
        pass

    def save_state(self, save_path):
        checkpoint = {"model": self.state_dict()}
        torch.save(checkpoint, save_path)

    def load_state(self, load_path):
        checkpoint = torch.load(load_path, map_location=self.device)
        self.load_state_dict(checkpoint['model'])
