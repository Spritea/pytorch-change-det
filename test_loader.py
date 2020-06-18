import sys, os
import torch
import argparse
import timeit
import numpy as np
import scipy.misc as misc
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from torch.utils import data
from tqdm import tqdm

from ptsemseg.models import get_model
from ptsemseg.loader import get_loader, get_data_path
from ptsemseg.utils import convert_state_dict

import yaml
from pathlib import Path
import natsort
import cv2 as cv

def test(args,cfg):

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_loader = get_loader(cfg['data']['dataset'])
    data_path = get_data_path(cfg['data']['dataset'],config_file=cfg)
    loader = data_loader(data_path, is_transform=True, img_norm=args.img_norm)
    n_classes = loader.n_classes

    t_loader = data_loader(
        data_path,
        is_transform=True,
        split='test',
        img_size=(cfg['data']['img_rows'], cfg['data']['img_cols']),)

    testloader = data.DataLoader(t_loader,
                                batch_size=1,
                                num_workers=cfg['training']['n_workers'])

    # Setup Model
    model = get_model(cfg['model'], n_classes)
    state = convert_state_dict(torch.load(args.model_path)["model_state"])
    # state=torch.load(args.model_path)["model_state"]
    model.load_state_dict(state)
    model.eval()
    model.to(device)

    with torch.no_grad():
        for i_val, (img_path,images_val) in tqdm(enumerate(testloader)):
            img_name=img_path[0]
            images_val = images_val.to(device)
            outputs = model(images_val)

            pred = np.squeeze(outputs.data.max(1)[1].cpu().numpy(), axis=0)
            decoded = loader.decode_segmap(pred)
            out_path="test_out/cdnet_change_det/"+Path(img_name).stem+".png"
            decoded_bgr = cv.cvtColor(decoded, cv.COLOR_RGB2BGR)
            # misc.imsave(out_path, decoded)
            cv.imwrite(out_path, decoded_bgr)

    # print("Classes found: ", np.unique(pred))
    # print("Segmentation Mask Saved at: {}".format(args.out_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Params")
    parser.add_argument(
        "--model_path",
        nargs="?",
        type=str,
        default="pretrain/cdnet_change_det/cdnet_my_6band_best_model.pkl",
        help="Path to the saved model",
    )
    parser.add_argument(
        "--img_norm",
        dest="img_norm",
        action="store_true",
        help="Enable input image scales normalization [0, 1] \
                              | True by default",)

    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/cdnet_change_det.yml",
        help="Configuration file to use"
    )
    args = parser.parse_args()
    with open(args.config) as fp:
        cfg = yaml.load(fp)
    test(args,cfg)
