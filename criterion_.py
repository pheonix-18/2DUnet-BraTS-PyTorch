import torch.nn as nn
import torch.functional as F
import monai
import torch
from medpy.metric.binary import hd95 as Hausdorff

def Dice(output, target, eps=1e-5):
    target = target.float()
    num = 2 * (output * target).sum()
    den = output.sum() + target.sum() + eps
    return 1.0 - num/den

def cal_Hausdoff(output, target, eps=1e-5):
    # Convert to One Hot Tensors
    _, labels = torch.max(output, dim = 1)
    # H0 = Hausdorff((b[...]==0).float(), (target == 0).float(), percentile=95)
    # H = Hausdorff((labels[...]).float().cpu(), target.float(), percentile=95)
    H1 = Hausdorff((labels[...]==1).float().cpu().numpy(), (target == 1).float().cpu().numpy())
    H2 = Hausdorff((labels[...]==2).float().cpu().numpy(), (target == 2).float().cpu().numpy())
    H3 = Hausdorff((labels[...]==3).float().cpu().numpy(), (target == 3).float().cpu().numpy())
    return H3, (H3 + H1)/2, (H3 + H2 + H1)/3

def dice_score(output, target, eps=1e-5):
    _, labels = torch.max(output, dim = 1)
    D0 = Dice((labels[...]==0).float(), (target == 0).float())
    D1 = Dice((labels[...]==1).float(), (target == 1).float())
    D2 = Dice((labels[...]==2).float(), (target == 2).float())
    D3 = Dice((labels[...]==3).float(), (target == 3).float())
    return 1-D0, 1-D1, 1-D2, 1-D3

class softmax_dice(nn.Module):
    '''
    The dice loss for using softmax activation function
    :param output: (b, num_class, d, h, w)
    :param target: (b, d, h, w)
    :return: softmax dice loss
    '''
    def __init__(self):
        super(softmax_dice, self).__init__()
        
    def forward(self, output, target):
        target[target == 4] = 3 
        output = output.cuda()
        target = target.cuda()
        loss0 = Dice(output[:, 0, ...], (target == 0).float())
        # import pdb
        # pdb.set_trace()
        loss1 = Dice(output[:, 1, ...], (target == 1).float())
        loss2 = Dice(output[:, 2, ...], (target == 2).float())
        loss3 = Dice(output[:, 3, ...], (target == 3).float())

        D0, D1, D2, D3 = dice_score(output, target)
        return loss1 + loss2 + loss3 + loss0, D0, D1, D2, D3
    
