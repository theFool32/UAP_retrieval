import os
import pickle

import torch
import torch.utils.data as data
from torch.autograd import Variable

from cirtorch.networks.imageretrievalnet import extract_vectors
from cirtorch.datasets.datahelpers import default_loader, imresize, cid2filename
from cirtorch.datasets.genericdataset import ImagesFromList
from cirtorch.examples.attack.myutil.utils import get_random_size
from cirtorch.utils.general import get_data_root
import random


class SfMDataset(data.Dataset):

    def __init__(self, name, imsize=None, transform=None, loader=default_loader, random_size=False):

        # setting up paths
        mode = 'val'
        data_root = get_data_root()
        db_root = os.path.join(data_root, 'train', name)
        ims_root = os.path.join(db_root, 'ims')

        # loading db
        db_fn = os.path.join(db_root, '{}.pkl'.format(name))
        with open(db_fn, 'rb') as f:
            db = pickle.load(f)[mode]

        # initializing tuples dataset
        self.name = name
        self.imsize = imsize
        self.images = [cid2filename(db['cids'][i], ims_root) for i in range(len(db['cids']))]
        self.clusters = db['cluster']
        self.qpool = db['qidxs']

        self.transform = transform
        self.loader = loader
        self.random_size = random_size

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            images tuple (q,p,n1,...,nN): Loaded train/val tuple at index of self.qidxs
        """
        if self.__len__() == 0:
            raise(RuntimeError("List qidxs is empty. Run ``dataset.create_epoch_tuples(net)`` method to create subset for train/val!"))

        # query image
        output = (self.loader(self.images[self.qpool[index]]))

        if self.imsize is not None:
            if self.random_size:
                w, h = output.size
                imsize = get_random_size(self.imsize, w, h)
                output = imresize(output, imsize)
            else:
                output = imresize(output, self.imsize)
        
        if self.transform is not None:
            output = self.transform(output)

        return output

    def __len__(self):
        return len(self.qpool)

