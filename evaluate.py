import torch
import torch.nn.functional as F
from tqdm import tqdm

import numpy as np

from utils.dice_score import multiclass_dice_coeff, dice_coeff
from utils.qscore import qScore


def evaluate(net, dataloader, device):
    net.eval()
    num_val_batches = len(dataloader)
    TPR , FPR, IoU = 0, 0, 0

    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        image, mask_true = batch['image'], batch['mask']
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            # predict the mask
            mask_pred = net(image)

            # convert to one-hot format
            if net.n_classes == 1:
                print("oi")
                # mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                # dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                # mask_pred_2 = mask_pred[:, 1:, ...]
                # print(np.min(mask_pred_2.cpu().numpy()))
                step = qScore(mask_pred[:, 1:, ...], mask_true[:, 1:, ...], reduce_batch_first=False)
                TPR += step[0]
                FPR += step[1]
                IoU += step[2]
                # dice_score += multiclass_dice_coeff(mask_pred_2, mask_true_2, reduce_batch_first=False)

           

    net.train()
    return TPR / num_val_batches, FPR / num_val_batches, IoU / num_val_batches
