
import numpy as np
import csv
from tutils import tdir, tfilename
# from utils import make_dir


class Evaluater(object):
    def __init__(self, logger, size):
        self.pixel_spaceing = 0.1

        self.logger = logger

        self.RE_list = list()

        self.recall_radius = [2, 2.5, 3, 4]  # 2mm etc
        self.recall_rate = list()

        self.Attack_RE_list = list()
        self.Defend_RE_list = list()

        self.dict_Attack = dict()
        self.dict_Defend = dict()
        self.total_list = dict()

        self.mode_list = [0, 1, 2, 3]
        self.mode_dict = {0: "Iterative FGSM", 1: "Adaptive Iterative FGSM", \
                          2: "Adaptive_Rate", 3: "Proposed"}
        # Just for hand dataset
        self.size = size  # original size is not fixed for hand dataset, we calculate it in realtime

        for mode in self.mode_list:
            self.dict_Defend[mode] = dict()
            self.dict_Attack[mode] = dict()
            self.total_list[mode] = list()

    def reset(self):
        self.RE_list.clear()
        for mode in self.mode_list:
            self.dict_Defend[mode] = dict()
            self.dict_Attack[mode] = dict()
            self.total_list[mode] = list()
        self.Attack_RE_list.clear()
        self.Defend_RE_list.clear()


    def record(self, pred, landmark):
        c = pred[0].shape[0]
        diff = np.zeros([c, 2], dtype=float)  # y, x
        for i in range(c):
            diff[i][0] = abs(pred[0][i] - landmark[i][1]) * self.scale_rate_y
            diff[i][1] = abs(pred[1][i] - landmark[i][0]) * self.scale_rate_x
        Radial_Error = np.sqrt(np.power(diff[:, 0], 2) + np.power(diff[:, 1], 2))
        Radial_Error *= self.pixel_spaceing
        self.RE_list.append(Radial_Error)
        return None

    def record_hand(self, pred, landmark, img_shape):
        """
        Function for testing hand dataset, (due to the different "img_shape")
        """
        # inverse the order of xy
        # for l in landmark:
        #     tmp = l[1]
        #     l[1] = l[0]
        #     l[0] = tmp

        scale_rate_y = img_shape[0] / self.size[0]
        scale_rate_x = img_shape[1] / self.size[1]
        c = pred[0].shape[0]
        diff = np.zeros([c, 2], dtype=float)  # y, x
        for i in range(c):
            diff[i][0] = abs(pred[0][i] - landmark[i][1]) * scale_rate_y
            diff[i][1] = abs(pred[1][i] - landmark[i][0]) * scale_rate_x
        Radial_Error = np.sqrt(np.power(diff[:, 0], 2) + np.power(diff[:, 1], 2))
        Radial_Error *= self.pixel_spaceing
        self.RE_list.append(Radial_Error)

    # def record_attack(self, pred, landmark, attack_list, mode=0, iteration=0):
    #     # n = batchsize = 1
    #     # pred : list[ c(y) ; c(x) ]
    #     # landmark: list [ (x , y) * c]
    #     assert (mode in [0, 1, 2, 3])
    #
    #     c = pred[0].shape[0]
    #     diff = np.zeros([c, 2], dtype=float)  # y, x
    #     attack_temp = list()
    #     defend_temp = list()
    #     for i in range(c):
    #         diff[i][0] = abs(pred[0][i] - landmark[i][1]) * self.scale_rate_y
    #         diff[i][1] = abs(pred[1][i] - landmark[i][0]) * self.scale_rate_x
    #         Radial_Error = np.sqrt(np.power(diff[i, 0], 2) + np.power(diff[i, 1], 2))
    #         if i in attack_list:
    #             attack_temp.append([i, Radial_Error * self.pixel_spaceing])
    #         else:
    #             defend_temp.append([i, Radial_Error * self.pixel_spaceing])
    #
    #     if iteration not in self.dict_Attack[mode].keys():
    #         self.dict_Attack[mode][iteration] = list()
    #     self.dict_Attack[mode][iteration].append(attack_temp)
    #     if iteration not in self.dict_Defend[mode].keys():
    #         self.dict_Defend[mode][iteration] = list()
    #     self.dict_Defend[mode][iteration].append(defend_temp)

    def cal_metrics(self):
        # calculate MRE SDR
        temp = np.array(self.RE_list)
        Mean_RE_channel = temp.mean(axis=0)
        # self.logger.info(Mean_RE_channel)
        with open('results.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(Mean_RE_channel.tolist())
        mre = Mean_RE_channel.mean()
        self.log("ALL MRE {}".format(mre))

        sdr_dict = {}
        for radius in self.recall_radius:
            total = temp.size
            shot = (temp < radius).sum()
            self.log("ALL SDR {}mm  {}".format \
                                 (radius, shot * 100 / total))
            sdr_dict[f"SDR {radius}"] = shot * 100 / total

        ret_dict = {'mre': mre}
        ret_dict = {**ret_dict, **sdr_dict}
        return ret_dict

    def log(self, msg, *args, **kwargs):
        if self.logger is None:
            print(msg, *args, **kwargs)
        else:
            self.logger.info(msg, *args, **kwargs)

