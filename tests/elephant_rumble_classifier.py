import os

import torch
from torch import nn


class ElephantRumbleClassifier(nn.Module):
    def __init__(
        self,
        input_dim=768,
        hidden_dim=768 // 4,
        output_dim=2,
        dropout=0.2,
    ):
        super().__init__()
        self.model_name = "[no weights loaded]"
        self.act = nn.LeakyReLU()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.dropout(x)
        x = self.linear1(x)
        x = self.act(x)
        x = self.dropout(x)
        return self.linear2(x)

    def get_cache_prefix(self):
        cache_prefix = torch.hub.get_dir()
        return os.path.join(cache_prefix, "fruitpunch_elephants")

    def choose_model_weights(self, model_name):
        if model_name in ["best", "training", "best_using_training_data_only"]:
            return "best.pth"
        if model_name in ["enhanced", "best_using_more_varied_training_data"]:
            return "best.pth"
        return model_name

    def load_pretrained_weights(self, model_name):
        if os.path.exists(model_name):
            self.load_state_dict(torch.load(model_name))
            self.eval()
            return
        if not model_name.endswith(".pth"):
            model_name = self.choose_model_weights(model_name)
            print(f"using {model_name}")
        self.model_name = model_name
        self.download_model_files_if_needed(model_name)
        cache_dir = self.get_cache_prefix()
        model_weights_file = os.path.join(cache_dir, model_name)
        self.load_state_dict(torch.load(model_weights_file))
        self.eval()

    def download_model_files_if_needed(self, pretrained_weights):
        src_prefix = "https://0ape.com/pretrained_models"
        dst_prefix = self.get_cache_prefix()
        src = f"{src_prefix}/{pretrained_weights}"
        dst = os.path.join(dst_prefix, pretrained_weights)
        if not os.path.exists(dst):
            print(f"fetching {dst} from {src}")
            os.makedirs(dst_prefix, exist_ok=True)
            torch.hub.download_url_to_file(src, dst)
