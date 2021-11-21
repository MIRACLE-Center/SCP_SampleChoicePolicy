import torch
import numpy as np
import argparse
from tutils import trans_args, trans_init, tfilename, save_script, CSVLogger
from tutils.trainer import DDPTrainer, LearnerModule, Monitor
from models.network import UNet_Pretrained
from datasets.ceph_basic import Cephalometric
from utils.tester import Tester


def focal_loss(pred, gt):
    return (-(1 - pred) * gt * torch.log(pred+1e-8) - pred * (1 - gt) * torch.log(1 - pred + 1e-8)).mean()


class Learner(LearnerModule):
    def __init__(self, logger, config, *args, **kwargs):
        super(Learner, self).__init__(*args, **kwargs)
        self.logger = logger
        self.config = config
        self.net = UNet_Pretrained(19, regression=False)
        self.loss_logic_fn = focal_loss
        self.config_training = config['training']

    def forward(self, x, **kwargs):
        return self.net(x)

    def training_step(self, data, batch_idx, **kwargs):
        img = data['img']
        mask = data['mask']
        heatmap = self.forward(img)
        loss = self.loss_logic_fn(heatmap, mask)
        if torch.isnan(loss):
            print("debug: ", img.shape, mask.shape, heatmap.shape, loss)
        return {'loss': loss}

    def configure_optimizers(self, **kwargs):
        optimizer = torch.optim.Adam(params=self.net.parameters(),
                                     lr=self.config_training['lr'],
                                     betas=(0.9, 0.999),
                                     eps=1e-08,
                                     weight_decay=self.config_training['weight_decay'],
                                     )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, \
                                                    self.config_training['decay_step'], \
                                                    gamma=self.config_training['decay_gamma'])
        return {'optimizer':optimizer, 'scheduler':scheduler}



def train(logger, config):
    model = Learner(logger=None, config=config)
    tester_subtest = Tester(logger=None, config=config, mode="subtest")
    monitor = Monitor(key='mre', mode='dec')
    dataset = Cephalometric(config['dataset']['pth'], return_offset=False)
    trainer = DDPTrainer(logger=logger, config=config, tester=tester_subtest, monitor=monitor)
    trainer.fit(model, dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Unet landmark detection network")
    parser.add_argument("--func", default="train")
    parser.add_argument("--config", default='configs/baseline/baseline.yaml', help="name of the run")
    parser.add_argument("--indices", type=str, default="")
    parser.add_argument("--epoch", type=int, default=0, help="default configs")
    parser.add_argument("--data", type=str, default="Train", help="Percentage of training data")
    parser.add_argument("--note", default="")
    args = trans_args(parser)
    logger, config = trans_init(args)
    save_script(config['base']['runs_dir'], __file__)
    # dump_yaml(logger, config)
    eval(args.func)(logger, config)

