import json
import math
import os
from typing import Optional

import torch
from datasets import load_dataset
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from my_utils.data_preprocessing import (
    FeatureType,
    ar_batch_preparation,
    preprocess_audio,
    set_pad_index,
)
from my_utils.encoding_convertions import krnParser
from networks.transformer.encoder import HEIGHT_REDUCTION, WIDTH_REDUCTION

SOS_TOKEN = "<SOS>"  # Start-of-sequence token
EOS_TOKEN = "<EOS>"  # End-of-sequence token


class ARDataModule(LightningDataModule):
    def __init__(
        self,
        ds_name: str,
        use_voice_change_token: bool = False,
        batch_size: int = 16,
        num_workers: int = 20,
        feature_type: FeatureType = "spectrogram",
    ):
        super(ARDataModule, self).__init__()
        self.ds_name = ds_name
        self.use_voice_change_token = use_voice_change_token
        self.batch_size = batch_size
        # self.num_workers = num_workers
        self.num_workers = 0  # TODO this is temprorary fix
        self.feature_type = feature_type

        # Datasets
        # To prevent executing setup() twice
        self.train_ds: Optional[ARDataset] = None
        self.val_ds: Optional[ARDataset] = None
        self.test_ds: Optional[ARDataset] = None

    def setup(self, stage: str):
        if stage == "fit":
            if not self.train_ds:
                self.train_ds = ARDataset(
                    ds_name=self.ds_name,
                    partition_type="train",
                    use_voice_change_token=self.use_voice_change_token,
                    feature_type=self.feature_type,
                )
            if not self.val_ds:
                self.val_ds = ARDataset(
                    ds_name=self.ds_name,
                    partition_type="val",
                    use_voice_change_token=self.use_voice_change_token,
                    feature_type=self.feature_type,
                )

        if stage == "test" or stage == "predict":
            if not self.test_ds:
                self.test_ds = ARDataset(
                    ds_name=self.ds_name,
                    partition_type="test",
                    use_voice_change_token=self.use_voice_change_token,
                )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=ar_batch_preparation,
        )  # prefetch_factor=2

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
        )  # prefetch_factor=2

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
        )  # prefetch_factor=2

    def predict_dataloader(self):
        print("Using test_dataloader for predictions.")
        return self.test_dataloader()

    def get_w2i_and_i2w(self):
        try:
            return self.train_ds.w2i, self.train_ds.i2w
        except AttributeError:
            return self.test_ds.w2i, self.test_ds.i2w

    def get_max_seq_len(self):
        try:
            return self.train_ds.max_seq_len
        except AttributeError:
            return self.test_ds.max_seq_len

    def get_max_audio_len(self):
        try:
            return self.train_ds.max_audio_len
        except AttributeError:
            return self.test_ds.max_audio_len


####################################################################################################

DATASETS = ["quartets", "beethoven", "mozart", "haydn"]
SPLITS = ["train", "val", "test"]

# split="train[:10%]+test[:10%:]+"
SUBSET_AMOUNT = "[:1%]"
SUBSET_AMOUNT = "[:1%]"  # what are u doing bro
SUBSET_AMOUNT = os.environ.get("SUBSET_AMOUNT", "")
FULL_SUBSETS = "".join([x + f"{SUBSET_AMOUNT}+" for x in SPLITS])[:-1]
# FULL_SUBSETS = "".join([x + "+" for x in SPLITS])[:-1]


class ARDataset(Dataset):
    def __init__(
        self,
        ds_name: str,
        partition_type: str,
        use_voice_change_token: bool = False,
        feature_type: FeatureType = "spectrogram",
    ):
        self.ds_name = ds_name.lower()
        self.partition_type = partition_type
        self.use_voice_change_token = use_voice_change_token
        self.feature_type = feature_type
        self.init(vocab_name="ar_w2i")
        self.max_seq_len += 1  # Add 1 for EOS_TOKEN

    def __len__(self):
        return len(self.ds)

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
        # if SUBSET_AMOUNT:
        self.ds = load_dataset(
            f"PRAIG/{self.ds_name}-quartets",
            split=f"{self.partition_type}{SUBSET_AMOUNT}",
            # split=f"{self.partition_type}",
            # streaming=True,
        )

        # Check and retrieve vocabulary
        # vocab_folder = os.path.join("Quartets", "vocabs")
        vocab_folder = "/scratch/22454862/Quartets/vocab"
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

        max_lens_folder = "/scratch/22454862/Quartets/max_lens"
        os.makedirs(max_lens_folder, exist_ok=True)
        max_lens_name = vocab_name
        self.max_lens_path = os.path.join(max_lens_folder, max_lens_name)
        max_lens = self.check_and_retrieve_max_lens()
        self.max_seq_len = max_lens["max_seq_len"]
        self.max_audio_len = max_lens["max_audio_len"]

    def __getitem__(self, idx):
        x = preprocess_audio(
            raw_audio=self.ds[idx]["audio"]["array"],
            sr=self.ds[idx]["audio"]["sampling_rate"],
            dtype=torch.float32,
            feature=self.feature_type,
        )
        y = self.preprocess_transcript(text=self.ds[idx]["transcript"])
        if self.partition_type == "train":
            return x, self.get_number_of_frames(x), y
        return x, y

    def preprocess_transcript(self, text: str):
        y = self.krn_parser.convert(text=text)
        y = [SOS_TOKEN] + y + [EOS_TOKEN]
        y = [self.w2i[w] for w in y]
        return torch.tensor(y, dtype=torch.int64)

    def transcript_generator(self):
        if SUBSET_AMOUNT:
            full_ds = load_dataset(f"PRAIG/{self.ds_name}-quartets", split=FULL_SUBSETS)
            yield from full_ds
            # for sample in full_ds:
            #     yield sample
        else:
            full_ds = load_dataset(f"PRAIG/{self.ds_name}-quartets", streaming=True)

            for split in SPLITS:
                yield from full_ds[split]
                # for sample in full_ds[split]:
                #     yield sample

    def make_vocabulary(self):
        print("Making ar vocabulary")

        vocab = []
        for data in self.transcript_generator():
            text = data["transcript"]
            transcript = self.krn_parser.convert(text=text)
            vocab.extend(transcript)
        vocab = [SOS_TOKEN, EOS_TOKEN] + vocab
        vocab = sorted(set(vocab))

        w2i = {}
        i2w = {}
        for i, w in enumerate(vocab):
            w2i[w] = i + 1
            i2w[i + 1] = w
        w2i["<PAD>"] = 0
        i2w[0] = "<PAD>"

        return w2i, i2w

    def get_number_of_frames(self, audio):
        # audio is the output of preprocess_audio
        # audio.shape = [1, freq_bins, time_frames]
        return math.ceil(audio.shape[1] / HEIGHT_REDUCTION) * math.ceil(
            audio.shape[2] / WIDTH_REDUCTION
        )

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
        print("Making max lengths")
        max_seq_len = 0
        max_audio_len = 0

        full_ds = load_dataset("PRAIG/quartets-quartets", split=FULL_SUBSETS)
        # for split in SPLITS:
        #     for sample in full_ds[split]:
        max_audio_raw = None
        max_duration = 0.0
        for i, sample in enumerate(full_ds):
            # Max transcript length
            transcript = self.krn_parser.convert(text=sample["transcript"])
            max_seq_len = max(max_seq_len, len(transcript))

            # dur = sample["audio"]["array"].shape[0] /
            # Max audio length
            print("raw audio shape", sample["audio"]["array"])
            audio = preprocess_audio(  # TODO this doenst have to be done we can just find longest thing without preprocessing then preprocess it later
                raw_audio=sample["audio"]["array"],
                sr=sample["audio"]["sampling_rate"],
                dtype=torch.float32,
                feature=self.feature_type,
            )
            max_audio_len = max(max_audio_len, audio.shape[2])

        return {
            "max_seq_len": max_seq_len,
            "max_audio_len": max_audio_len,
        }

    # def iterate_dataset(self):
    #     if SUBSET_AMOUNT:
