import json
import math
import os

import torch
from datasets import load_dataset
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from my_utils.data_preprocessing import (
    ctc_batch_preparation,
    preprocess_audio,
    set_pad_index,
)
from my_utils.encoding_convertions import krnParser

DATASETS = ["quartets", "beethoven", "mozart", "haydn"]
SPLITS = ["train", "val", "test"]

# split="train[:10%]+test[:10%:]+"
SUBSET_AMOUNT = "[:1%]"
FULL_SUBSETS = "".join([x + f"{SUBSET_AMOUNT}+" for x in SPLITS])[:-1]


class CTCDataset(Dataset):
    def __init__(
        self,
        ds_name: str,
        partition_type: str,
        width_reduction: int = 2,
        use_voice_change_token: bool = False,
    ):
        self.ds_name = ds_name.lower()
        self.partition_type = partition_type
        self.width_reduction = width_reduction
        self.use_voice_change_token = use_voice_change_token
        self.init(vocab_name="ctc_w2i")

    def init(self, vocab_name: str = "w2i"):
        # Initialize krn parser
        self.krn_parser = krnParser(use_voice_change_token=self.use_voice_change_token)

        # Check dataset name
        assert self.ds_name in DATASETS, f"Invalid dataset name: {self.ds_name}"

        # Check partition type
        assert self.partition_type in SPLITS, (
            f"Invalid partition type: {self.partition_type}"
        )

        # Get audios and transcripts files
        self.ds = load_dataset(
            f"PRAIG/{self.ds_name}-quartets",
            split=f"{self.partition_type}{SUBSET_AMOUNT}",
        )

        # Check and retrieve vocabulary
        vocab_folder = os.path.join("Quartets", "vocabs")
        os.makedirs(vocab_folder, exist_ok=True)
        vocab_name = self.ds_name + f"_{vocab_name}"
        vocab_name += "_withvc" if self.use_voice_change_token else ""
        vocab_name += ".json"
        self.w2i_path = os.path.join(vocab_folder, vocab_name)
        self.w2i, self.i2w = self.check_and_retrieve_vocabulary()
        # Modify the global PAD_INDEX to match w2i["<PAD>"]
        set_pad_index(self.w2i["<PAD>"])

        # Check and retrive max lengths
        # Set max_seq_len, max_audio_len and frame_multiplier_factor
        max_lens_folder = os.path.join("Quartets", "max_lens")
        os.makedirs(max_lens_folder, exist_ok=True)
        max_lens_name = vocab_name
        self.max_lens_path = os.path.join(max_lens_folder, max_lens_name)
        max_lens = self.check_and_retrieve_max_lens()
        self.max_seq_len = max_lens["max_seq_len"]
        self.max_audio_len = max_lens["max_audio_len"]
        self.frame_multiplier_factor = max_lens["max_frame_multiplier_factor"]

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        x = preprocess_audio(
            raw_audio=self.ds[idx]["audio"]["array"],
            sr=self.ds[idx]["audio"]["sampling_rate"],
            dtype=torch.float32,
        )
        y = self.preprocess_transcript(text=self.ds[idx]["transcript"])
        if self.partition_type == "train":
            # x.shape = [channels, height, width]
            return (
                x,
                (x.shape[2] // self.width_reduction)
                * self.width_reduction
                * self.frame_multiplier_factor,
                y,
                len(y),
            )
        return x, y

    def preprocess_transcript(self, text: str):
        y = self.krn_parser.convert(text=text)
        y = [self.w2i[w] for w in y]
        return torch.tensor(y, dtype=torch.int32)

    def check_and_retrieve_vocabulary(self):
        w2i = {}
        i2w = {}

        if os.path.isfile(self.w2i_path):
            with open(self.w2i_path, "r") as file:
                w2i = json.load(file)
            i2w = {v: k for k, v in w2i.items()}
        else:
            w2i, i2w = self.make_vocabulary()
            with open(self.w2i_path, "w") as file:
                json.dump(w2i, file)

        return w2i, i2w

    def make_vocabulary(self):
        print("Making ctc vocab")
        full_ds = load_dataset(f"PRAIG/{self.ds_name}-quartets", split=FULL_SUBSETS)

        vocab = []
        # for split in SPLITS:
        # for text in full_ds[split]["transcript"]:
        for i, text in enumerate(full_ds["transcript"]):
            if i % 50 == 0:
                print(f"Making vocabulary, item: {i}")
            transcript = self.krn_parser.convert(text=text)
            vocab.extend(transcript)
        vocab = sorted(set(vocab))

        w2i = {}
        i2w = {}
        for i, w in enumerate(vocab):
            w2i[w] = i + 1
            i2w[i + 1] = w
        w2i["<PAD>"] = 0
        i2w[0] = "<PAD>"

        return w2i, i2w

    def check_and_retrieve_max_lens(self):
        max_lens = {}

        if os.path.isfile(self.max_lens_path):
            with open(self.max_lens_path, "r") as file:
                max_lens = json.load(file)
        else:
            max_lens = self.make_max_lens()
            with open(self.max_lens_path, "w") as file:
                json.dump(max_lens, file)

        return max_lens

    def make_max_lens(self):
        # Set the maximum lengths for the whole QUARTETS collection:
        # 1) Get the maximum transcript length
        # 2) Get the maximum audio length
        # 3) Get the frame multiplier factor so that
        # the frames input to the RNN are equal to the
        # length of the transcript, ensuring the CTC condition
        print("Making max lengths")
        max_seq_len = 0
        max_audio_len = 0
        max_frame_multiplier_factor = 0

        full_ds = load_dataset("PRAIG/quartets-quartets", split=FULL_SUBSETS)
        # for split in SPLITS:
        #     for sample in full_ds[split]:
        for i, sample in enumerate(full_ds):
            if i % 50 == 0:
                print(f"Making max lens, item: {i}")
            # Max transcript length
            transcript = self.krn_parser.convert(text=sample["transcript"])
            max_seq_len = max(max_seq_len, len(transcript))

            # Max audio length
            audio = preprocess_audio(
                raw_audio=sample["audio"]["array"],
                sr=sample["audio"]["sampling_rate"],
                dtype=torch.float32,
            )
            max_audio_len = max(max_audio_len, audio.shape[2])
            # Max frame multiplier factor
            max_frame_multiplier_factor = max(
                max_frame_multiplier_factor,
                math.ceil(((2 * len(transcript)) + 1) / audio.shape[2]),
            )

        return {
            "max_seq_len": max_seq_len,
            "max_audio_len": max_audio_len,
            "max_frame_multiplier_factor": max_frame_multiplier_factor,
        }
