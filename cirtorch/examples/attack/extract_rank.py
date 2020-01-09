import argparse
import os
import shutil
import time
import math
import pickle
import pdb
import random

import numpy as np

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.datasets as datasets

from cirtorch.networks.imageretrievalnet import init_network, extract_vectors
from cirtorch.layers.loss import ContrastiveLoss
from cirtorch.datasets.datahelpers import collate_tuples, cid2filename
from cirtorch.datasets.traindataset import TuplesDataset
from cirtorch.datasets.testdataset import configdataset
from cirtorch.utils.download import download_train, download_test
from cirtorch.utils.whiten import whitenlearn, whitenapply
from cirtorch.utils.evaluate import compute_map_and_print
from cirtorch.utils.general import get_data_root, htime
from cirtorch.examples.attack.myutil.utils import do_whiten
from cirtorch.examples.attack.myutil.triplet_dataset import MyTripletDataset
from cirtorch.datasets.genericdataset import ImagesFromList

training_dataset_names = ["retrieval-SfM-120k", "Landmarks"]
test_datasets_names = [
    "oxford5k,paris6k",
    "roxford5k,rparis6k",
    "oxford5k,paris6k,roxford5k,rparis6k",
]
test_whiten_names = ["retrieval-SfM-30k", "retrieval-SfM-120k"]

model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)
pool_names = ["mac", "spoc", "gem", "rmac"]

parser = argparse.ArgumentParser(description="PyTorch CNN Image Retrieval Training")

parser.add_argument(
    "--network-path", help="network path, destination where network is saved"
)
parser.add_argument(
    "--image-size",
    default=362,
    type=int,
    metavar="N",
    help="maximum size of longer image side used for training (default: 1024)",
)

# standard train/val options
parser.add_argument(
    "--gpu-id",
    "-g",
    default="0",
    metavar="N",
    help="gpu id used for training (default: 0)",
)
parser.add_argument(
    "--workers",
    "-j",
    default=8,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 8)",
)
parser.add_argument(
    "--epochs",
    default=100,
    type=int,
    metavar="N",
    help="number of total epochs to run (default: 100)",
)
min_loss = float("inf")


def main():
    global args, min_loss
    args = parser.parse_args()

    # set cuda visible device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    # set random seeds (maybe pass as argument)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    np.random.seed(1234)
    random.seed(1234)
    torch.backends.cudnn.deterministic = True

    state = torch.load(args.network_path)
    model = init_network(
        model=state["meta"]["architecture"],
        pooling=state["meta"]["pooling"],
        whitening=state["meta"]["whitening"],
        mean=state["meta"]["mean"],
        std=state["meta"]["std"],
        pretrained=False,
    )
    model.load_state_dict(state["state_dict"])
    model.meta["Lw"] = state["meta"]["Lw"]
    model.cuda()

    # whitening
    Lw = model.meta["Lw"]["retrieval-SfM-120k"]["ss"]
    Lw_m = torch.from_numpy(Lw["m"]).cuda().float()
    Lw_p = torch.from_numpy(Lw["P"]).cuda().float()

    # Data loading code
    normalize = transforms.Normalize(mean=model.meta["mean"], std=model.meta["std"])
    transform = transforms.Compose([transforms.ToTensor(), normalize])
    query_dataset = MyTripletDataset(
        # imsize=args.image_size,
        imsize=(362, 362),
        random=False,
        transform=transform,
        norm=(0, 0),
        filename="base/" + args.network_path.replace("/", "_") + "_triplet",
    )
    # val_dataset.test_cluster(model)
    # return
    query_dataset.create_epoch_tuples(model)
    query_loader = torch.utils.data.DataLoader(
        query_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        worker_init_fn=lambda _: random.seed(1234),
    )

    base_dataset = ImagesFromList(
        root="",
        images=query_dataset.images,
        # imsize=query_dataset.imsize,
        imsize=(362, 362),
        transform=query_dataset.transform,
        random=False,
    )
    base_loader = torch.utils.data.DataLoader(base_dataset, batch_size=1, shuffle=False)

    # test(["oxford5k"], model)
    extract(query_loader, base_loader, model, Lw_m, Lw_p)


def extract(query_loader, base_loader, model, Lw_m, Lw_p):
    # create tuples for validation
    query_loader.dataset.create_epoch_tuples(model)

    # switch to evaluate mode
    model.eval()

    nq = len(query_loader.dataset)
    nb = len(base_loader.dataset)
    base_features = torch.Tensor(model.meta["outputdim"], nb).cuda()
    query_features = torch.Tensor(model.meta["outputdim"], nq).cuda()
    ranks = torch.Tensor(nq, nb)
    network_path = args.network_path.replace("/", "_")
    os.makedirs(f"ranks/{network_path}", exist_ok=True)
    with torch.no_grad():
        print(">>> base")
        for i, input in enumerate(base_loader):
            feature = model(input.cuda())
            feature = do_whiten(feature, Lw_m, Lw_p)
            base_features[:, i] = feature.squeeze()
        print(">>> base over")
        torch.save(base_features, f"ranks/{network_path}/base_362")

        print(">>> query")
        for i, input in enumerate(query_loader):
            feature = model(input.cuda())
            feature = do_whiten(feature, Lw_m, Lw_p)
            query_features[:, i] = feature.squeeze()
            score = base_features.t() @ feature
            _, rank = torch.sort(score, dim=0, descending=True)
            ranks[i, :] = rank.squeeze()
        torch.save(query_features, f"ranks/{network_path}/query_362")
        print(">>> query over")
    torch.save(ranks, f"ranks/{network_path}/ranks_362")


def test(datasets, net):
    print(">> Evaluating network on test datasets...")
    image_size = 1024

    # moving network to gpu and eval mode
    net.cuda()
    net.eval()
    # set up the transform
    normalize = transforms.Normalize(mean=net.meta["mean"], std=net.meta["std"])
    transform = transforms.Compose([transforms.ToTensor(), normalize])

    # compute whitening
    # Lw = None
    Lw = net.meta["Lw"]["retrieval-SfM-120k"]["ss"]

    # evaluate on test datasets
    # datasets = args.test_datasets.split(",")
    for dataset in datasets:
        start = time.time()

        print(">> {}: Extracting...".format(dataset))

        # prepare config structure for the test dataset
        cfg = configdataset(dataset, os.path.join(get_data_root(), "test"))
        images = [cfg["im_fname"](cfg, i) for i in range(cfg["n"])]
        qimages = [cfg["qim_fname"](cfg, i) for i in range(cfg["nq"])]
        bbxs = [tuple(cfg["gnd"][i]["bbx"]) for i in range(cfg["nq"])]

        # extract database and query vectors
        print(">> {}: database images...".format(dataset))
        vecs = extract_vectors(net, images, image_size, transform)
        print(">> {}: query images...".format(dataset))
        qvecs = extract_vectors(net, qimages, image_size, transform, bbxs)

        print(">> {}: Evaluating...".format(dataset))

        # convert to numpy
        vecs = vecs.numpy()
        qvecs = qvecs.numpy()

        # search, rank, and print
        scores = np.dot(vecs.T, qvecs)
        ranks = np.argsort(-scores, axis=0)
        compute_map_and_print(dataset, ranks, cfg["gnd"])

        if Lw is not None:
            # whiten the vectors
            vecs_lw = whitenapply(vecs, Lw["m"], Lw["P"])
            qvecs_lw = whitenapply(qvecs, Lw["m"], Lw["P"])

            # search, rank, and print
            scores = np.dot(vecs_lw.T, qvecs_lw)
            ranks = np.argsort(-scores, axis=0)
            compute_map_and_print(dataset + " + whiten", ranks, cfg["gnd"])

        print(">> {}: elapsed time: {}".format(dataset, htime(time.time() - start)))


def save_checkpoint(state, is_best, directory):
    filename = os.path.join(directory, "model_epoch%d.pth.tar" % state["epoch"])
    torch.save(state, filename)
    if is_best:
        filename_best = os.path.join(directory, "model_best.pth.tar")
        shutil.copyfile(filename, filename_best)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == "__main__":
    main()
