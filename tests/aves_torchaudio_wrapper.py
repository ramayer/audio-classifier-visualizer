from __future__ import annotations

import json
import os
from typing import Any

import torch
import torch.hub
from torchaudio.models import wav2vec2_model


class AvesTorchaudioWrapper(torch.nn.Module):
    def __init__(
        self,
        config: str = "aves-base-bio.torchaudio.model_config.json",
        weights: str = "aves-base-bio.torchaudio.pt",
    ) -> None:
        super().__init__()
        self.download_model_files_if_needed([config, weights])
        cachedir = self.get_cache_prefix()
        config_path = os.path.join(cachedir, config)
        weights_path = os.path.join(cachedir, weights)
        self.config = self.load_config(config_path)
        self.model = wav2vec2_model(**self.config, aux_num_out=None)
        self.model.load_state_dict(torch.load(weights_path))
        self.model.feature_extractor.requires_grad_(False)
        self.eval()

    def load_config(self, config_path: str) -> dict[str, Any]:
        with open(config_path) as ff:
            return json.load(ff)

    def forward(self, sig: torch.Tensor) -> torch.Tensor:
        return self.model.extract_features(sig)[0][-1]

    def get_cache_prefix(self) -> str:
        cache_prefix = torch.hub.get_dir()
        return os.path.join(cache_prefix, "fruitpunch_elephants")

    def download_model_files_if_needed(self, required_files: list[str]) -> None:
        src_prefix = "https://storage.googleapis.com/esp-public-files/ported_aves"
        dst_prefix = self.get_cache_prefix()
        for f in required_files:
            src = f"{src_prefix}/{f}"
            dst = os.path.join(dst_prefix, f)
            if not os.path.exists(dst):
                print(f"fetching {dst} from {src}")
                os.makedirs(dst_prefix, exist_ok=True)
                torch.hub.download_url_to_file(src, dst)
