import os
import pickle
import pdb
from random import sample
import random
import time

import numpy as np
import torch
import torch.utils.data as data
from torch.autograd import Variable

from cirtorch.networks.imageretrievalnet import extract_vectors
from cirtorch.datasets.datahelpers import default_loader, imresize, cid2filename
from cirtorch.datasets.genericdataset import ImagesFromList
from cirtorch.utils.general import get_data_root
from sklearn.cluster import KMeans
from cirtorch.examples.attack.myutil.utils import get_random_size
from cirtorch.examples.attack.myutil.utils import do_whiten

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class Distillation_dataset(data.Dataset):
    def __init__(
        self,
        imsize=None,
        nnum=5,
        qsize=2000,
        poolsize=20000,
        transform=None,
        loader=default_loader,
        filename=None,
        q_percent=1,
    ):

        # setting up paths
        data_root = get_data_root()
        name = "retrieval-SfM-120k"
        db_root = os.path.join(data_root, "train", name)
        ims_root = os.path.join(db_root, "ims")

        # loading db
        db_fn = os.path.join(db_root, "{}.pkl".format(name))
        with open(db_fn, "rb") as f:
            db = pickle.load(f)["val"]

        # initializing tuples dataset
        self.imsize = imsize
        self.images = [
            cid2filename(db["cids"][i], ims_root) for i in range(len(db["cids"]))
        ]
        self.clusters = db["cluster"]
        self.qpool = db["qidxs"]
        # self.ppool = db['pidxs']

        # size of training subset for an epoch
        self.nnum = nnum
        self.qsize = min(qsize, len(self.qpool))
        self.poolsize = min(poolsize, len(self.images))
        self.qidxs = self.qpool
        self.index = np.arange(len(self.qidxs))

        if q_percent < 1:
            number = int(len(self.qidxs) * q_percent)
            self.index = np.random.permutation(self.index)
            self.index = self.index[:number]

        self.pidxs = []
        self.nidxs = []

        self.poolvecs = None

        self.transform = transform
        self.loader = loader
        self.filename = filename
        self.phase = 1

        self.ranks = torch.load(f"{filename}/ranks_362")
        self.pool_vecs = pickle.load(open(f"{filename}/pool_vecs", "rb"))
        print(len(self.images))
        self.loaded_images = []
        if os.path.exists("./images"):
            self.loaded_images = pickle.load(open("./images", "rb"))
        else:
            for i in range(len(self.images)):
                try:
                    img = self.loader(self.images[i])
                    if self.imsize is not None:
                        img = imresize(img, self.imsize)
                    if self.transform is not None:
                        img_tensor = self.transform(img).unsqueeze(0)
                    img.close()
                    self.loaded_images.append(img_tensor)
                except:
                    self.loaded_images.append(None)
            pickle.dump(self.loaded_images, open("./images", "wb"))

    def __getitem__(self, index):
        # Not used
        # coding in ../distillation.py
        pass

    def __len__(self):
        return len(self.index)
        if not self.qidxs:
            return 0
        return len(self.qidxs)
