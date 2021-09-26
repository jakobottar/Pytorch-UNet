import numpy as np
import skimage.measure as skiM
from torch import Tensor

def countNumRegs(img):

	pCount = skiM.label(img, background = 0) + 1	
	if np.min(pCount) != 0:
		pCount -= 1
	return pCount


def qScore(prediction: Tensor, label: Tensor, reduce_batch_first: bool = False):

    assert prediction.size() == label.size()

    prediction = prediction.cpu().numpy().squeeze((0,1))
    label = label.cpu().numpy().squeeze((0,1))

    if reduce_batch_first:
        raise ValueError(f'qScore: idk mate')

    else:
        pCount = countNumRegs(prediction)	
        lCount = countNumRegs(label)

        lCount_flat = np.copy(lCount).reshape(lCount.shape[0]*lCount.shape[1],1)
        pCount_flat = np.copy(pCount).reshape(pCount.shape[0]*pCount.shape[1],1)

        # Unique regions
        valUniqueP = np.unique(pCount); valUniqueP = valUniqueP[np.where(valUniqueP > 0)]
        valUniqueL = np.unique(lCount); valUniqueL = valUniqueL[np.where(valUniqueL > 0)]

        overlapT = 0.6
        countP = 0; countN = 0

        trueposLabel = np.zeros((label.shape))
        trueposPredict = np.copy(trueposLabel)

        for i in range(len(valUniqueP)):
            indPredict = np.where(pCount_flat == valUniqueP[i])[0]			
            
            ## Find the region values of true label corresponding with current predict indices
            temp = np.unique(lCount_flat[indPredict])			
            temp = temp[np.where(temp > 0)]

            for j in range(len(temp)):
                
                indTrue = np.where(lCount_flat == temp[j])[0]
                
                inter = np.intersect1d(indPredict,indTrue)
                un = np.union1d(indPredict,indTrue)

                if len(inter) >= overlapT*len(un):
                    countP += 1

                    ## Obtain true positive regions for both ground truth and predict
                    trueposLabel[np.where(lCount == temp[j])] = 1
                    trueposPredict[np.where(pCount == valUniqueP[i])] = 1
                else:
                    countN += 1

        
        # imgTP = np.copy(trueposPredict)
        # imgTP_label = np.copy(trueposLabel)

        trueposPredict = trueposPredict.reshape(trueposPredict.shape[0]*trueposPredict.shape[1],1)
        indPosPredict = np.where(trueposPredict > 0)[0]	
        trueposLabel = trueposLabel.reshape(trueposLabel.shape[0]*trueposLabel.shape[1],1)
        indPosLabel = np.where(trueposLabel > 0)[0]		

        '''f, ax = plt.subplots(1,3)
        ax[0].imshow(label); ax[1].imshow(prediction); ax[2].imshow(posLabel)'''
        numPosLabel = len(valUniqueL)		
        numPosPredict = len(valUniqueP)
        
        TPR = 0.0
        if numPosLabel > 0.0:
            TPR = countP / numPosLabel
        
        FPR = 0.0
        if numPosPredict != 0.0:
            FPR = (numPosPredict-countP) / numPosPredict

        IoU = 0.0
        if len(np.union1d(indPosLabel,indPosPredict)) > 0.0:
            IoU = len(np.intersect1d(indPosPredict,indPosLabel)) \
                / len(np.union1d(indPosLabel,indPosPredict))

        return TPR , FPR, IoU