import argparse
import math
import os
import pdb
import pickle
import random
import shutil
import time
from pprint import pprint

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

from cirtorch.datasets.datahelpers import cid2filename, collate_tuples
from cirtorch.datasets.testdataset import configdataset
from cirtorch.datasets.traindataset import TuplesDataset
from cirtorch.examples.attack.myutil.baseline import result as baseline_result
from cirtorch.examples.attack.myutil.mi_sgd import (MI_SGD, SIGN_AdaBound,
                                                    SIGN_Adam)
from cirtorch.examples.attack.myutil.sfm_dataset import SfMDataset
from cirtorch.examples.attack.myutil.triplet_dataset import MyTripletDataset
from cirtorch.examples.attack.myutil.utils import (MultiLoss, bcolors,
                                                   do_whiten, idcg, inv_gfr,
                                                   one_hot, rescale_check)
from cirtorch.layers.loss import ContrastiveLoss
from cirtorch.networks.imageretrievalnet import extract_vectors, init_network
from cirtorch.utils.download import download_test, download_train
from cirtorch.utils.evaluate import compute_map_and_print
from cirtorch.utils.general import get_data_root, htime
from cirtorch.utils.whiten import whitenapply, whitenlearn

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

base = {}  # storing the feature of base
MAX_EPS = 10.0 / 255  # max eps of perturbation
MODE = "bilinear"  # mode of resize

parser = argparse.ArgumentParser(description="PyTorch CNN Image Retrieval Training")

# export directory, training and val datasets, test datasets
parser.add_argument(
    "--test-datasets",
    "-td",
    metavar="DATASETS",
    default="oxford5k,paris6k,roxford5k,rparis6k",
    choices=test_datasets_names,
    help="comma separated list of test datasets: "
    + " | ".join(test_datasets_names)
    + " (default: oxford5k,paris6k)",
)
parser.add_argument(
    "--network-path", help="network path, destination where network is saved"
)
parser.add_argument(
    "--image-size",
    default=1024,
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
    default=1,
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
parser.add_argument(
    "--batch-size",
    "-b",
    default=1,
    type=int,
    metavar="N",
    help="number of (q,p,n1,...,nN) tuples in a mini-batch (default: 5)",
)
parser.add_argument(
    "--print-freq",
    default=500,
    type=int,
    metavar="N",
    help="print frequency (default: 10)",
)
parser.add_argument("--noise-path", type=str, help="noise path")
parser.add_argument(
    "--loss-margin",
    "-lm",
    metavar="LM",
    default=0.8,
    type=float,
    help="loss margin: (default: 0.7)",
)
parser.add_argument(
    "--image-size-L", default=256, type=int, help="min of image size for random"
)
parser.add_argument(
    "--image-size-H", default=1024, type=int, help="max of image size for random"
)
parser.add_argument("--noise-size", default=1024, type=int, help="noise-size")

# loss
parser.add_argument(
    "--point_wise", dest="point_wise", action="store_true", help="point-wise loss"
)
parser.add_argument(
    "--label_wise", dest="label_wise", action="store_true", help="label-wise loss"
)
parser.add_argument(
    "--pair_wise", dest="pair_wise", action="store_true", help="pair-wise loss"
)
parser.add_argument(
    "--list_wise", dest="list_wise", action="store_true", help="list_wise loss"
)

parser.add_argument("--max-eps", default=10, type=int, help="max eps")

args = parser.parse_args()
pprint(args)


def main():
    global base
    global MAX_EPS
    MAX_EPS = args.max_eps / 255.0

    # load base
    fname = args.network_path.replace("/", "_") + ".pkl"
    if os.path.exists(f"base/{fname}"):
        with open(f"base/{fname}", "rb") as f:
            base = pickle.load(f)

    # for saving noise
    os.makedirs(args.noise_path, exist_ok=True)

    # set cuda visible device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    np.random.seed(1234)
    random.seed(1234)
    torch.backends.cudnn.deterministic = True

    # load retrieval model
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

    # perturbation for training
    noise = torch.zeros((3, args.noise_size, args.noise_size)).cuda()

    print(state["meta"]["architecture"])
    print(state["meta"]["pooling"])
    noise.requires_grad = True

    optimizer = MI_SGD(
        [
            {"params": [noise], "lr": MAX_EPS / 10, "momentum": 1, "sign": True},
            # {"params": [noise], "lr": 1e-2, "momentum": 1, "sign": True},
        ],
        max_eps=MAX_EPS,
    )
    print(optimizer)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=math.exp(-0.01))

    # Data loading code
    normalize = transforms.Normalize(mean=model.meta["mean"], std=model.meta["std"])
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            # normalize,
        ]
    )
    mean = torch.Tensor(normalize.mean).view(3, 1, 1)
    std = torch.Tensor(normalize.std).view(3, 1, 1)

    # dataloader
    val_dataset = MyTripletDataset(
        imsize=(args.image_size_L, args.image_size_H),
        transform=transform,
        norm=(mean, std),
        filename="base/" + args.network_path.replace("/", "_") + "_triplet",
    )
    val_dataset.create_epoch_tuples(model)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        worker_init_fn=lambda _: random.seed(1234),
    )

    # load classifier model
    if args.label_wise:
        classification_model = torch.load(
            "base/" + args.network_path.replace("/", "_") + "_triplet.KMeans_cls.pth"
        )
    else:
        classification_model = None

    noise_best = None
    min_loss = float("inf")
    min_epoch = -1
    for epoch in range(args.epochs):
        # set manual seeds per epoch
        np.random.seed(epoch + 1234)
        torch.manual_seed(epoch + 1234)
        torch.cuda.manual_seed_all(epoch + 1234)
        random.seed(epoch + 1234)

        # train for one epoch on train set
        scheduler.step()
        begin_time = time.time()
        loss, noise = train(
            val_loader,
            model,
            noise,
            epoch,
            normalize,
            classification_model,
            optimizer,
            None,
        )
        print("epoch time", time.time() - begin_time)

        # evaluate on test datasets
        loss = test(args.test_datasets, model, noise.cpu(), 1024)
        print(bcolors.str(f"test fgr: {1-loss}", bcolors.OKGREEN))

        # remember best loss and save checkpoint
        is_best = loss < min_loss
        min_loss = min(loss, min_loss)
        save_noise(noise, is_best, epoch)
        if is_best:
            min_epoch = epoch
            noise_best = noise.clone().detach()
        if epoch - min_epoch > 5:
            break

    print("Best")
    loss = test(args.test_datasets, model, noise_best.cpu(), 1024)
    print(bcolors.str(f"test fgr: {1-loss}", bcolors.OKGREEN))


def train(train_loader, model, noise, epoch, normalize, cls, optimizer, multiLoss):
    """ train perturbation
    train_loader: data loader
    model: victim retrieval model
    noise: perturbation to be optimized
    epoch: current epoch
    normalize: data normalize parameter
    cls: classification model
    optimizer: optimizer for iter
    multiLoss: multi loss
    """

    global args
    noise.requires_grad = True
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.eval()

    # normalize
    mean = normalize.mean
    std = normalize.std
    mean = torch.Tensor(mean).view(1, 3, 1, 1).cuda()
    std = torch.Tensor(std).view(1, 3, 1, 1).cuda()

    # whitening
    Lw = model.meta["Lw"]["retrieval-SfM-120k"]["ss"]
    Lw_m = torch.from_numpy(Lw["m"]).cuda().float()
    Lw_p = torch.from_numpy(Lw["P"]).cuda().float()

    # cluster center and base cluster id
    pool_clusters_centers = train_loader.dataset.pool_clusters_centers.cuda().float()
    clustered_pool = train_loader.dataset.clustered_pool

    end = time.time()
    optimizer.zero_grad()
    optimizer.rescale()
    for i, (input) in enumerate(train_loader):
        # measure data loading time.
        data_time.update(time.time() - end)
        model.zero_grad()

        input = input.cuda()
        with torch.no_grad():
            norm_output = (input - mean) / std
            feature = model(norm_output)
            feature = do_whiten(feature, Lw_m, Lw_p).detach()

        optimizer.zero_grad()
        current_noise = noise
        current_noise = F.interpolate(
            current_noise.unsqueeze(0),
            mode=MODE,
            size=tuple(input.shape[-2:]),
            align_corners=True,
        ).squeeze()
        perturted_input = torch.clamp(input + current_noise, 0, 1)

        perturbed_input = (perturted_input - mean) / std
        perturbed_feature = model(perturbed_input)
        perturbed_feature = do_whiten(perturbed_feature, Lw_m, Lw_p)

        # pair-wise
        if args.pair_wise:
            with torch.no_grad():
                scores = torch.mm((pool_clusters_centers), feature)
                scores, ranks = torch.sort(scores, dim=0, descending=True)

                pos_i = ranks[0, 0].item()
                neg_i = ranks[-1, 0].item()
                # neg_feature = torch.from_numpy(
                #     np.concatenate(
                #         (
                #             clustered_pool[neg_i][
                #                 np.random.choice(clustered_pool[neg_i].shape[0]), :
                #             ].reshape(1, -1),
                #         )
                #     )
                # ).cuda()
                # pos_feature = (
                #     torch.from_numpy(
                #         clustered_pool[pos_i][
                #             np.random.choice(clustered_pool[pos_i].shape[0]), :
                #         ]
                #     )
                #     .cuda()
                #     .unsqueeze(0)
                # )
            neg_feature = pool_clusters_centers[neg_i, :].view(1, -1)
            pos_feature = pool_clusters_centers[pos_i, :].view(1, -1)
            perturbed_feature = perturbed_feature.t()
            # neg_feature = torch.cat((neg_feature, -feature.t()))
            # pos_feature = torch.cat((pos_feature, feature.t()))
            # perturbed_feature = torch.cat(
            #     (perturbed_feature.t(), perturbed_feature.t())
            # )
            neg_feature = neg_feature * 10
            pos_feature = pos_feature * 10
            perturbed_feature = perturbed_feature * 10

            pair_loss = F.triplet_margin_loss(
                perturbed_feature, neg_feature, pos_feature, args.loss_margin
            )
        else:
            pair_loss = torch.zeros(1).cuda()

        # point-wise
        if args.point_wise:
            point_loss = (
                torch.dot(perturbed_feature.squeeze(), feature.squeeze()) + 1
            ) / 2
        else:
            point_loss = torch.zeros(1).cuda()

        # label-wise
        if args.label_wise:
            actual_pred = cls(feature.t())
            perturbed_pred = cls(perturbed_feature.t())
            actual_label = actual_pred.max(1, keepdim=True)[1].item()
            one_hot_actual_label = one_hot(
                perturbed_pred.size(1), torch.LongTensor([actual_label]).cuda()
            ).float()
            label_loss = F.relu(
                (perturbed_pred * one_hot_actual_label).sum()
                - (perturbed_pred * (1 - one_hot_actual_label)).max()
            )
        else:
            label_loss = torch.zeros(1).cuda()

        if args.list_wise:
            clean_scores = torch.mm((pool_clusters_centers), feature)
            _, clean_ranks = torch.sort(clean_scores, dim=0, descending=True)

            # pos_i = clean_ranks[:256, :].squeeze()
            # neg_i = clean_ranks[256:, :].squeeze()
            pos_i = clean_ranks[:, :].squeeze()
            neg_i = torch.flip(pos_i, (0,))

            scores = -torch.mm((pool_clusters_centers), perturbed_feature)
            _, ranks = torch.sort(scores, dim=0, descending=True)

            doc_ranks = torch.zeros(pool_clusters_centers.size(0)).to(feature.device)
            doc_ranks[ranks] = 1 + torch.arange(pool_clusters_centers.size(0)).to(
                feature.device
            ).float().view((-1, 1))
            doc_ranks = doc_ranks.view((-1, 1))

            score_diffs = scores[pos_i] - scores[neg_i].view(neg_i.size(0))
            exped = score_diffs.exp()
            N = 1 / idcg(pos_i.size(0))
            ndcg_diffs = (1 / (1 + doc_ranks[pos_i])).log2() - (
                1 / (1 + doc_ranks[neg_i])
            ).log2().view(neg_i.size(0))

            lamb_updates = -1 / (1 + exped) * N * ndcg_diffs.abs()
            lambs = torch.zeros((pool_clusters_centers.shape[0], 1)).to(feature.device)
            lambs[pos_i] += lamb_updates.sum(dim=1, keepdim=True)
            lambs[neg_i] -= lamb_updates.sum(dim=0, keepdim=True).t()
            scores.backward(lambs)
            list_loss = torch.zeros(1).cuda()
        else:
            list_loss = torch.zeros(1).cuda()

        label_loss = label_loss.view(1)
        point_loss = point_loss.view(1)
        pair_loss = pair_loss.view(1)
        list_loss = list_loss.view(1)

        loss = label_loss + point_loss + pair_loss

        if not args.list_wise:
            loss.backward()

        losses.update(loss.item())
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            # optimizer.rescale()
            print(
                ">> Train: [{0}][{1}/{2}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Noise l2: {noise:.4f}".format(
                    epoch + 1,
                    i,
                    len(train_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    noise=noise.norm(),
                )
            )

    noise.requires_grad = False
    print(bcolors.str(f"Train {epoch}: Loss: {losses.avg}", bcolors.OKGREEN))
    return losses.avg, noise


def test(datasets, net, noise, image_size):
    global base
    print(">> Evaluating network on test datasets...")

    net.cuda()
    net.eval()
    normalize = transforms.Normalize(mean=net.meta["mean"], std=net.meta["std"])

    def add_noise(img):
        n = noise
        n = F.interpolate(
            n.unsqueeze(0), mode=MODE, size=tuple(img.shape[-2:]), align_corners=True
        ).squeeze()
        return torch.clamp(img + n, 0, 1)

    transform_base = transforms.Compose([transforms.ToTensor(), normalize])
    transform_query = transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(add_noise), normalize]
    )

    if "Lw" in net.meta:
        Lw = net.meta["Lw"]["retrieval-SfM-120k"]["ss"]
    else:
        Lw = None

    # evaluate on test datasets
    datasets = args.test_datasets.split(",")
    attack_result = {}
    for dataset in datasets:
        start = time.time()

        print(">> {}: Extracting...".format(dataset))

        cfg = configdataset(dataset, os.path.join(get_data_root(), "test"))
        images = [cfg["im_fname"](cfg, i) for i in range(cfg["n"])]
        qimages = [cfg["qim_fname"](cfg, i) for i in range(cfg["nq"])]
        bbxs = [tuple(cfg["gnd"][i]["bbx"]) for i in range(cfg["nq"])]

        # extract database and query vectors
        print(">> {}: database images...".format(dataset))
        with torch.no_grad():
            if dataset in base and str(image_size) in base[dataset]:
                vecs = base[dataset][str(image_size)]
            else:
                vecs = extract_vectors(net, images, image_size, transform_base)
                if dataset not in base:
                    base[dataset] = {}
                base[dataset][str(image_size)] = vecs
                fname = args.network_path.replace("/", "_") + ".pkl"
                with open(f"base/{fname}", "wb") as f:
                    pickle.dump(base, f)
            print(">> {}: query images...".format(dataset))
            qvecs = extract_vectors(net, qimages, image_size, transform_query, bbxs)

        print(">> {}: Evaluating...".format(dataset))

        # convert to numpy
        vecs = vecs.numpy()
        qvecs = qvecs.numpy()

        # whiten the vectors
        vecs_lw = whitenapply(vecs, Lw["m"], Lw["P"])
        qvecs_lw = whitenapply(qvecs, Lw["m"], Lw["P"])

        # search, rank, and print
        scores = np.dot(vecs_lw.T, qvecs_lw)
        ranks = np.argsort(-scores, axis=0)
        r = compute_map_and_print(dataset + " + whiten", ranks, cfg["gnd"])
        attack_result[dataset] = r

        print(">> {}: elapsed time: {}".format(dataset, htime(time.time() - start)))
    return inv_gfr(
        attack_result, baseline_result[net.meta["architecture"]][net.meta["pooling"]]
    )


def save_noise(noise, is_best, epoch):
    filename = os.path.join(args.noise_path, "noise_%d" % epoch)
    np.save(filename, noise.cpu().numpy())
    torchvision.utils.save_image(noise, filename + ".png", normalize=True)
    if is_best:
        filename_best = os.path.join(args.noise_path, "noise_best")
        shutil.copyfile(filename + ".npy", filename_best + ".npy")
        shutil.copyfile(filename + ".png", filename_best + ".png")


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
