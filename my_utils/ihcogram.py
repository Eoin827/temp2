import sys

import numpy as np
import torch
import yaml

sys.path.append("./IHCApproxNH/")
from classes import WaveNet
from utils import utils

with open("./IHCApproxNH/config/config31rfa3-1fullSet.yaml", "r") as ymlfile:
    conf = yaml.safe_load(ymlfile)  #

    # Constants
    sigMax = torch.tensor(55)
    ihcogramMax = torch.tensor(1.33)
    ihcogramMax = utils.comp(ihcogramMax, conf["scaleWeight"], conf["scaleType"])
    fs = 16000

    # number of samples to be skipped due to WaveNet processing
    skipLength = (2 ** conf["nLayers"]) * conf["nStacks"]

    ## initialize WaveNet and load model paramaters
    NET = WaveNet.WaveNet(
        conf["nLayers"],
        conf["nStacks"],
        conf["nChannels"],
        conf["nResChannels"],
        conf["nSkipChannels"],
        conf["numOutputLayers"],
    )

    NET.load_state_dict(
        torch.load(
            "./IHCApproxNH/model/musan31rfa3-1fullSet_20231014-145738.pt",
            map_location=torch.device("cuda:0"),
            weights_only=True,
        )
    )

    # Freeze the IHC layers
    for param in NET.parameters():
        param.requires_grad = False


def forward(inputs: np.ndarray):
    # signals, _ = torch.nn.utils.rnn.pad_packed_sequence(inputs, batch_first=True)
    # signals = signals.permute(0, 2, 1)

    signals = (
        torch.from_numpy(inputs).to(torch.float32).to("cuda:0")
    )  # TODO idk why people are hardcoding i will do it for now  as well ig

    signals = torch.from_numpy(inputs).to(
        "cuda:0"
    )  # TODO idk why people are hardcoding i will do it for now  as well ig
    with torch.no_grad():
        IHC_predicted = NET(signals)

        IHC_predicted = IHC_predicted * ihcogramMax
        return utils.invcomp(IHC_predicted, conf["scaleWeight"], conf["scaleType"])
