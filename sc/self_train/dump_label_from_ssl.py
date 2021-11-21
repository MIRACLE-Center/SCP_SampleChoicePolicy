import torch
import numpy as np
from tutils import trans_args, trans_init, dump_yaml, tfilename, save_script
import argparse
from utils.tester_ssl import Tester
from models.network_emb_study import UNet_Pretrained



def dump_labels_from_ssl(logger, config, model):
    tester = Tester(logger, config)
    tester.test(model, dump_label=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ssl/ssl.yaml")
    parser.add_argument("--experiment", default="dump_labels")
    args = trans_args(parser)
    logger, config = trans_init(args, file=__file__)

    ckpt = config['training']['ckpt'] = "/home1/quanquan/code/landmark/code/runs/ssl/ssl/debug2/ckpt/best_model_epoch_890.pth"
    model = UNet_Pretrained(3, emb_len=config['training']['emb_len'], non_local=config['training']['non_local'])
    state_dict = torch.load(ckpt)
    model.load_state_dict(state_dict)
    model.cuda()

    dump_labels_from_ssl(logger, config, model)
    dump_yaml(logger, config)