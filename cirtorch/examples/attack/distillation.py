import argparse
import os
import shutil
import time
import math
import pickle
import pdb

import numpy as np

import torch
import torch.multiprocessing

# torch.multiprocessing.set_sharing_strategy("file_system")
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
from cirtorch.examples.attack.myutil.distillation_dataset import Distillation_dataset
from cirtorch.examples.attack.myutil.utils import bcolors
from cirtorch.examples.attack.myutil.utils import do_whiten
from cirtorch.layers.normalization import L2N

f = os.path.realpath(__file__)
f = open(f, "r")
print("".join(f.readlines()))
f.close()

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
loss_names = ["contrastive", "cross_entropy"]
optimizer_names = ["sgd", "adam"]

parser = argparse.ArgumentParser(description="PyTorch CNN Image Retrieval Training")

# export directory, training and val datasets, test datasets
parser.add_argument(
    "directory", metavar="DIR", help="destination where trained network should be saved"
)
parser.add_argument(
    "--target", help="destination where trained network should be saved"
)
parser.add_argument(
    "--test-datasets",
    "-td",
    metavar="DATASETS",
    default="oxford5k",
    choices=test_datasets_names,
    help="comma separated list of test datasets: "
    + " | ".join(test_datasets_names)
    + " (default: oxford5k,paris6k)",
)
parser.add_argument(
    "--test-whiten",
    metavar="DATASET",
    default="",
    choices=test_whiten_names,
    help="dataset used to learn whitening for testing: "
    + " | ".join(test_whiten_names)
    + " (default: None)",
)

# network architecture and initialization options
parser.add_argument(
    "--arch",
    "-a",
    metavar="ARCH",
    default="resnet101",
    choices=model_names,
    help="model architecture: " + " | ".join(model_names) + " (default: resnet101)",
)
parser.add_argument(
    "--pool",
    "-p",
    metavar="POOL",
    default="gem",
    choices=pool_names,
    help="pooling options: " + " | ".join(pool_names) + " (default: gem)",
)
parser.add_argument(
    "--whitening",
    "-w",
    dest="whitening",
    action="store_true",
    help="train model with end-to-end whitening",
)
parser.add_argument(
    "--not-pretrained",
    dest="pretrained",
    action="store_false",
    help="use model with random weights (default: pretrained on imagenet)",
)
parser.add_argument(
    "--loss",
    "-l",
    metavar="LOSS",
    default="contrastive",
    choices=loss_names,
    help="training loss options: " + " | ".join(loss_names) + " (default: contrastive)",
)
parser.add_argument(
    "--loss-margin",
    "-lm",
    metavar="LM",
    default=0.7,
    type=float,
    help="loss margin: (default: 0.7)",
)

# train/val options specific for image retrieval learning
parser.add_argument(
    "--image-size",
    default=362,
    type=int,
    metavar="N",
    help="maximum size of longer image side used for training (default: 1024)",
)
parser.add_argument(
    "--query-size",
    "-qs",
    default=2000,
    type=int,
    metavar="N",
    help="number of queries randomly drawn per one train epoch (default: 2000)",
)
parser.add_argument(
    "--pool-size",
    "-ps",
    default=20000,
    type=int,
    metavar="N",
    help="size of the pool for hard negative mining (default: 20000)",
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
    default=500,
    type=int,
    metavar="N",
    help="number of total epochs to run (default: 100)",
)
parser.add_argument(
    "--batch-size",
    "-b",
    default=10,
    type=int,
    metavar="N",
    help="number of (q,p,n1,...,nN) tuples in a mini-batch (default: 5)",
)
parser.add_argument(
    "--optimizer",
    "-o",
    metavar="OPTIMIZER",
    default="adam",
    choices=optimizer_names,
    help="optimizer options: " + " | ".join(optimizer_names) + " (default: adam)",
)
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=5e-7,
    type=float,
    metavar="LR",
    help="initial learning rate (default: 1e-6)",
)
parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
parser.add_argument(
    "--weight-decay",
    "--wd",
    default=1e-4,
    type=float,
    metavar="W",
    help="weight decay (default: 1e-4)",
)
parser.add_argument(
    "--print-freq",
    default=50,
    type=int,
    metavar="N",
    help="print frequency (default: 10)",
)
parser.add_argument(
    "--is_pretrained", dest="is_pretrained", action="store_true", help="is_pretrained"
)
parser.add_argument(
    "--is_random", dest="is_random", action="store_true", help="is_random"
)
parser.add_argument("--notion", help="notion")
parser.add_argument("--q_percent", default=1, type=float)

min_loss = float("inf")


class Whiten_layer(nn.Module):
    def __init__(self, d_in, d_out):
        super(Whiten_layer, self).__init__()
        self.w = nn.Linear(d_in, d_out)
        self.norm = L2N()

    def forward(self, x):
        return self.norm(self.w(x))


def main():
    global args, min_loss
    args = parser.parse_args()
    print(args)

    # set cuda visible device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    # check if test dataset are downloaded
    # and download if they are not
    download_train(get_data_root())
    download_test(get_data_root())

    # create export dir if it doesnt exist
    directory = "{}".format(args.target.replace("/", "_"))
    directory += "_{}".format(args.arch)
    directory += "_{}".format(args.pool)
    if args.whitening:
        directory += "_whiten"
    if not args.pretrained:
        directory += "_notpretrained"
    # directory += "_bsize{}_imsize{}".format(args.batch_size, args.image_size)
    directory += "_pretrained" if args.is_pretrained else ""
    directory += "_random" if args.is_random else ""
    directory += args.notion

    target_net = args.target[6:].replace("_", "/")
    print(target_net)
    state = torch.load(target_net)
    lw = state["meta"]["Lw"]["retrieval-SfM-120k"]["ss"]

    args.directory = os.path.join(args.directory, directory)
    print(">> Creating directory if it does not exist:\n>> '{}'".format(args.directory))
    if not os.path.exists(args.directory):
        os.makedirs(args.directory)

    # set random seeds (maybe pass as argument)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)

    # create model
    print(">> Using pre-trained model '{}'".format(args.arch))
    model = init_network(
        model=args.arch,
        pooling=args.pool,
        whitening=args.whitening,
        pretrained=not args.is_random,
    )
    model.cuda()

    target_model = init_network(
        model=args.arch,
        pooling=args.pool,
        whitening=args.whitening,
        pretrained=not args.is_random,
    )
    target_model.load_state_dict(state["state_dict"])
    lw_m = lw["m"].copy()
    lw_p = lw["P"].copy()
    target_model.lw_m = nn.Parameter(torch.from_numpy(lw_m).float())
    target_model.lw_p = nn.Parameter(torch.from_numpy(lw_p).float())
    target_model.cuda()

    lw_m = lw["m"].copy()
    lw_p = lw["P"].copy()
    model.lw_m = nn.Parameter(torch.from_numpy(lw_m).float())
    model.lw_p = nn.Parameter(torch.from_numpy(lw_p).float())
    whiten_layer = Whiten_layer(lw["P"].shape[1], lw["P"].shape[0])
    model.white_layer = whiten_layer
    model.cuda()

    # parameters split into features and pool (no weight decay for pooling layer)
    parameters = [
        {"params": model.features.parameters()},
        {"params": model.pool.parameters(), "lr": args.lr * 10, "weight_decay": 0},
        {"params": model.white_layer.parameters(), "lr": 1e-2, "weight_decay": 5e-1},
    ]
    if model.whiten is not None:
        parameters.append({"params": model.whiten.parameters()})

    # define optimizer
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            parameters, args.lr, momentum=args.momentum, weight_decay=args.weight_decay
        )
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            parameters, args.lr, weight_decay=args.weight_decay
        )

    # define learning rate decay schedule
    exp_decay = math.exp(-0.01)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=exp_decay)

    # optionally resume from a checkpoint
    start_epoch = 0
    # Data loading code
    normalize = transforms.Normalize(mean=model.meta["mean"], std=model.meta["std"])
    transform = transforms.Compose([transforms.ToTensor(), normalize])
    val_dataset = Distillation_dataset(
        imsize=(args.image_size, args.image_size),
        nnum=1,
        qsize=float("Inf"),
        poolsize=float("Inf"),
        transform=transform,
        filename=args.target,
        q_percent=args.q_percent,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=10,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False,
        collate_fn=collate_tuples,
    )

    min_epoch = -1
    for epoch in range(start_epoch, args.epochs):
        if args.is_pretrained or args.is_random:
            break

        # set manual seeds per epoch
        np.random.seed(epoch)
        torch.manual_seed(epoch)
        torch.cuda.manual_seed_all(epoch)

        # adjust learning rate for each epoch
        scheduler.step()

        loss = train(val_loader, model, optimizer, epoch, target_model)
        print(loss)

        # evaluate on test datasets
        if (epoch + 1) % 1 == 0:
            with torch.no_grad():
                test(args.test_datasets, model, lw)

        # remember best loss and save checkpoint
        is_best = loss < min_loss
        min_loss = min(loss, min_loss)

        save_checkpoint(
            {
                "epoch": epoch + 1,
                "meta": model.meta,
                "state_dict": model.state_dict(),
                "min_loss": min_loss,
                "optimizer": optimizer.state_dict(),
            },
            is_best,
            args.directory,
        )
        if is_best:
            min_epoch = epoch
        # if epoch - min_epoch > 5:
        #     # break
        #     if val_dataset.phase == 1:
        #         print(bcolors.str(">>> phase 2", bcolors.OKGREEN))
        #         val_dataset.phase = 2
        #         min_epoch = epoch
        #         for group in optimizer.param_groups:
        #             group["lr"] /= 10
        #     else:
        #         break

    if args.is_pretrained or args.is_random:
        save_checkpoint(
            {
                "epoch": 0 + 1,
                "meta": model.meta,
                "state_dict": model.state_dict(),
                "min_loss": min_loss,
                "optimizer": optimizer.state_dict(),
            },
            True,
            args.directory,
        )
    # print("calculate whiten")
    # lw = learning_lw(model)
    # filename = os.path.join(args.directory, "lw")
    # pickle.dump(lw, open(filename, "wb"))


def train(train_loader, model, optimizer, epoch, target_model):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    dataset = train_loader.dataset
    l = np.arange(len(dataset))
    print(len(l))
    # np.random.shuffle(l)
    optimizer.zero_grad()
    end = time.time()
    for batch_i, index in enumerate(l):
        data_time.update(time.time() - end)
        end = time.time()
        r = dataset.ranks[index, :].long().numpy()
        BIN = 128
        if dataset.phase == 1:
            size = len(r) // BIN
            bid = []
            for i in range(BIN):
                bi = np.random.choice(r[i * size : i * size + size], 1)[0]
                while dataset.loaded_images[bi] is None:
                    bi = np.random.choice(r[i * size : i * size + size], 1)[0]
                bid.append(bi)
        elif dataset.phase == 2:
            # For convenience
            bid = r[:BIN]

        output = []
        output.append(dataset.loaded_images[dataset.qidxs[index]])
        for bi in bid:
            output.append(dataset.loaded_images[bi].detach())
        output = torch.cat(output).cuda()

        target_output = target_model(output)
        target_output = do_whiten(target_output, target_model.lw_m, target_model.lw_p)

        while True:
            my_output = model(output)
            my_output = model.white_layer(my_output.t()).t()
            similarity = my_output[:, 0].view(1, -1) @ my_output[:, 1:]
            diff = similarity.t() - similarity - (1e-1 if dataset.phase == 1 else 1e-6)
            diff = -diff.triu() - torch.eye(BIN).cuda()
            coff = [i for i in range(diff.size(0) - 1, 0, -1)] + [0]
            diff = diff * torch.Tensor(coff).view(-1, 1).cuda()
            loss = nn.functional.relu(diff)
            loss = loss.sum() - 10000 * similarity.std()

            losses.update(loss.item(), 1)
            loss.backward()
            batch_time.update(time.time() - end)
            end = time.time()
            if (batch_i + 0) % args.batch_size == 0:
                optimizer.step()

            if batch_i % args.print_freq == 0:
                print(
                    ">> Train: [{0}][{1}/{2}]\t"
                    "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})".format(
                        epoch + 1,
                        batch_i,
                        len(dataset),
                        batch_time=batch_time,
                        data_time=data_time,
                        loss=losses,
                    )
                )
            break
    optimizer.step()

    return losses.avg


def learning_lw(net):
    net.cuda()
    net.eval()
    # set up the transform
    normalize = transforms.Normalize(mean=net.meta["mean"], std=net.meta["std"])
    transform = transforms.Compose([transforms.ToTensor(), normalize])

    test_whiten = "retrieval-SfM-30k"
    print(">> {}: Learning whitening...".format(test_whiten))

    # loading db
    db_root = os.path.join(get_data_root(), "train", test_whiten)
    ims_root = os.path.join(db_root, "ims")
    db_fn = os.path.join(db_root, "{}-whiten.pkl".format(test_whiten))
    with open(db_fn, "rb") as f:
        db = pickle.load(f)
    images = [cid2filename(db["cids"][i], ims_root) for i in range(len(db["cids"]))]

    # extract whitening vectors
    print(">> {}: Extracting...".format(args.test_whiten))
    wvecs = extract_vectors(net, images, 1024, transform)

    # learning whitening
    print(">> {}: Learning...".format(args.test_whiten))
    wvecs = wvecs.numpy()
    m, P = whitenlearn(wvecs, db["qidxs"], db["pidxs"])
    Lw = {"m": m, "P": P}
    return Lw


def test(datasets, net, lw):

    print(">> Evaluating network on test datasets...")

    # for testing we use image size of max 1024
    image_size = 1024

    # moving network to gpu and eval mode
    net.cuda()
    net.eval()
    # set up the transform
    normalize = transforms.Normalize(mean=net.meta["mean"], std=net.meta["std"])
    transform = transforms.Compose([transforms.ToTensor(), normalize])

    Lw = lw
    Lw = None

    # evaluate on test datasets
    datasets = args.test_datasets.split(",")
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

        vecs = do_whiten(vecs.cuda(), net.lw_m, net.lw_p).cpu()
        qvecs = do_whiten(qvecs.cuda(), net.lw_m, net.lw_p).cpu()

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


def set_batchnorm_eval(m):
    classname = m.__class__.__name__
    if classname.find("BatchNorm") != -1:
        # freeze running mean and std
        m.eval()
        # freeze parameters
        # for p in m.parameters():
        # p.requires_grad = False


if __name__ == "__main__":
    main()
