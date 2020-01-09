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

FNAME = "base/pool_vgg_gem"
TIMES = 1


class MyTripletDataset(data.Dataset):
    def __init__(
        self,
        imsize=None,
        nnum=5,
        qsize=2000,
        poolsize=20000,
        transform=None,
        loader=default_loader,
        norm=None,
        filename=None,
        random=True,
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
        self.qidxs = None
        self.pidxs = None
        self.nidxs = None

        self.poolvecs = None

        self.transform = transform
        self.loader = loader
        self.pool_clusters_centers = None
        self.clustered_pool = []
        self.norm = norm
        self.kmeans_ = None
        if filename is None:
            self.filename = FNAME
        else:
            self.filename = filename

        self.loaded_imgs = []
        self.random = random

    def __getitem__(self, index):
        # output = self.loader(self.images[self.qidxs[index]])
        output = self.loaded_imgs[index]

        if self.imsize is not None:
            if self.random:
                w, h = output.size
                imsize = get_random_size(self.imsize, w, h)
                output = imresize(output, imsize)
            else:
                output = imresize(output, self.imsize)

        if self.transform is not None:
            output = self.transform(output)
        return output

    def __len__(self):
        if not self.qidxs:
            return 0
        return len(self.qidxs)

    def create_epoch_tuples(self, net):

        print(">> Creating tuples...")
        if not os.path.exists(self.filename) or not os.path.exists(
            self.filename + ".KMeans"
        ):
            self.cluster(net)

        self.qidxs = self.qpool
        self.pidxs = []
        self.nidxs = []

        # prepare network
        net.cuda()
        net.eval()

        print(">> Extracting descriptors for pool...")
        fname = self.filename
        print("")

        print("cluster...")
        fname = fname + ".KMeans"
        with open(fname, "rb") as f:
            p = pickle.load(f)
        self.kmeans_ = p["kmeans"]
        self.clustered_pool = p["clustered_pool"]
        self.pool_clusters_centers = torch.from_numpy(self.kmeans_.cluster_centers_)
        print("")

        # for idx in self.qidxs:
        #     self.loaded_imgs.append(self.loader(self.images[idx]))
        # pickle.dump(self.loaded_imgs, open("data/train_imgs.pkl", "wb"))
        self.loaded_imgs = pickle.load(open("data/train_imgs.pkl", "rb"))

    def cluster(self, net):
        self.pidxs = []
        self.nidxs = []

        # draw poolsize random images for pool of negatives images
        idxs2images = torch.randperm(len(self.images))[: self.poolsize]

        # prepare network
        self.net = net
        net.cuda()
        net.eval()

        Lw = net.meta["Lw"]["retrieval-SfM-120k"]["ss"]
        Lw_m = torch.from_numpy(Lw["m"]).cuda().float()
        Lw_p = torch.from_numpy(Lw["P"]).cuda().float()

        print(">> Extracting descriptors for pool...")
        loader = torch.utils.data.DataLoader(
            ImagesFromList(
                root="",
                images=[self.images[i] for i in idxs2images],
                imsize=self.imsize,
                transform=self.transform,
                random=True,
            ),
            batch_size=1,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        )
        fname = self.filename
        if os.path.exists(fname):
            self.poolvecs = torch.load(fname).cuda()
        else:
            self.poolvecs = torch.Tensor(
                net.meta["outputdim"], len(idxs2images) * TIMES
            ).cuda()
            with torch.no_grad():
                for _ in range(TIMES):
                    print(_)
                    for i, input in enumerate(loader):
                        print(
                            "\r>>>> {}/{} done...".format(i + 1, len(idxs2images)),
                            end="",
                        )
                        input = (input - self.norm[0]) / self.norm[1]
                        b = net(Variable(input.cuda()))
                        b = do_whiten(b, Lw_m, Lw_p)
                        self.poolvecs[:, i + _ * len(idxs2images)] = b.squeeze()
            torch.save(self.poolvecs.cpu(), fname)
        print("")

        print(">> KMeans...")
        poolvecs = self.poolvecs.cpu().numpy().T
        fname = fname + ".KMeans"
        kmeans = KMeans(n_clusters=512, n_jobs=-1)
        kmeans.fit(poolvecs)
        clustered_pool = []
        self.pool_clusters_centers = torch.from_numpy(kmeans.cluster_centers_)
        for i in range(kmeans.cluster_centers_.shape[0]):
            clustered_pool.append(poolvecs[kmeans.labels_ == i, :])
        with open(fname, "wb") as f:
            pickle.dump({"kmeans": kmeans, "clustered_pool": clustered_pool}, f)
