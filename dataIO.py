## In this file we handle all data input and output for the model

import os
import random
import skimage.measure as skiM
import skimage.morphology as skiMore
import numpy as np
import progressbar
from PIL import Image, ImageOps

import torch
from torchvision import transforms
from torch.utils import data

def randomPatch(image_width, image_height, patch_width, patch_height):
    x1, y1 = 0, 0
    if image_width-patch_width != 0 or image_height-patch_height != 0:
        x1 = random.randrange(0, image_width-patch_width)
        y1 = random.randrange(0, image_height-patch_height)

    return (x1, y1, x1+patch_width, y1+patch_height)

# Extract color pixels as segmented pixels
# TODO: Is there a faster way to do this? this takes a long time
def makeMask(img):
    img_arr = np.array(img)

    r = img_arr[:, :, 0].reshape(img_arr.shape[0], img_arr.shape[1])
    g = img_arr[:, :, 1].reshape(img_arr.shape[0], img_arr.shape[1])
    b = img_arr[:, :, 2].reshape(img_arr.shape[0], img_arr.shape[1])		

    temp1 = np.zeros((r.shape)); temp1[np.where(r != g)] = 1
    temp2 = np.zeros((r.shape)); temp2[np.where(b != g)] = 1
    temp3 = np.zeros((r.shape)); temp3[np.where(r != b)] = 1

    t = temp1 + temp2 + temp3
    labelImg = np.zeros((r.shape))
    labelImg[np.where(t > 1)] = 255

    if True:
        temp = np.zeros((r.shape)); temp[np.where(g == 255)] = 1			
        labelImg[np.where(temp > 0)] = 0					
        
    return Image.fromarray(labelImg).convert(mode="RGB")

# Remove regions near boundaries
def boundaryFilt(image):

	label = (np.array(image) / 255).astype('uint8')

	topCut = 2; endCut = -1
	pLabel = np.copy(label)
	
	temp = np.copy(label)
	temp[:,:topCut] = np.max(label)
	temp[:,endCut:] = np.max(label)
	temp[:topCut,:] = np.max(label)
	temp[endCut:,:] = np.max(label)	

	pCount = skiM.label(temp, background = 0) + 1	
	if np.min(pCount) != 0:
		pCount -= 1

	pCount[np.where(pCount == 1)] = 0

	pLabel[np.where(pCount == 0)] = 0
	
	return Image.fromarray(pLabel * 255)

## Preprocessing of images
""" 
    This function preprocesses a set of images and corresponding masks for training.
    It reads in the images and masks and performs random crops and rotations on the images
    args:
        image_source: the source filepath for the image files
        mask_source: the source filepath for the corresponding masks
        num_patches: how many random patches of the image we should crop
        height: height of cropped patches
        width: width of cropped patches
        num_ang: number of 90-degree rotations to generate
            rotations are in order, 0, 90, 180, 270
"""
def Preprocess(image_files, mask_files, num_patches, height, width, num_ang, train_file_location = "./data/train/"):
    ext_length = 4 # length of the file extension

    # get list if images and masks from chosen folders

    widgets=[
          f"preprocessing raw images... ",
          progressbar.Bar(), 
          progressbar.Percentage()  
        ] 
    bar = progressbar.ProgressBar(max_value=len(image_files), widgets=widgets)

    for i, (img_file, mask_file) in enumerate(zip(image_files, mask_files)):
        filename = img_file.split('/')[-1][:-ext_length] # get filename
        # ext = img_file.split('/')[-1][-ext_length:] # get file extension
        ext = ".png"
        
        # load individual image files
        img_raw = Image.open(img_file)
        msk_raw = Image.open(mask_file)

        # our raw images contain a legend at the bottom which needs to be cropped off
        img = img_raw.crop((0,0,1024,880))
        msk = msk_raw.crop((0,0,1024,880))

        # convert the mask color image to a binary mask
        msk = makeMask(msk)

        # create and save paths for saving mask and image files
        image_path = os.path.join(train_file_location, "images")
        mask_path = os.path.join(train_file_location, "masks")

        try:
            os.makedirs(image_path) # create image folder if it does not exist
        except FileExistsError: True
        try:
            os.makedirs(mask_path) # create mask folder if it does not exist
        except FileExistsError: True

        # get random patches and rotate them
        for p in range(num_patches):
            # generate a random patch of the original image 
            # and crop image and mask with same patch
            img_w, img_h = img.size
            patch = randomPatch(img_w, img_h, width, height)
            img_crop = img.crop(patch)
            msk_crop = msk.crop(patch)

            for a in range(num_ang):
                # rotate images and masks
                ang = (a % 4) * 90
                img_rot = img_crop.rotate(ang)			
                msk_rot = msk_crop.rotate(ang)

                msk_filt = boundaryFilt(msk_rot)

                # generate filenames for image and mask
                img_name = os.path.join(image_path, f"uo3_{i}_{p}_{a}.jpg")
                mask_name = os.path.join(mask_path, f"uo3_{i}_{p}_{a}_mask.gif")

                # save files
                img_rot.save(img_name) 
                msk_filt.save(mask_name) 

        bar.update(i) # iterate progress bar

    bar.finish() # end progress bar

## Custom Dataloader for particle seperation dataset
"""
    This class is an extension of the torch.utils.data.Dataset class 
    and serves as a helper to get and read the image and mask files
    args:
        images: a list of image filenames, ex ["U3O8_fromUO4_Rep3_001_preprocessed_0_0.png",...]
        masks: a list of mask filenames, ex ["U3O8_fromUO4_Rep3_001_preprocessed_0_0.png",...]
        root: the location of the training dataset, ex "./data/train/"
            note that image files are expected in 'root'/images/ and 
            masks in 'root'/masks/
        transform: transformations to be applied to image files upon loading 
            NOTE: NOT APPLIED TO MASK
"""
class SegmentationDataset(data.Dataset):
    def __init__(self, images: list, masks: list, root, transform=None):
        self.root = root
        self.images = images
        self.masks = masks
        self.transform = transform
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # open image and mask
        image = ImageOps.grayscale(Image.open(os.path.join(self.root, "images", self.images[idx])))
        mask = ImageOps.grayscale(Image.open(os.path.join(self.root, "masks", self.masks[idx])))

        # if transform is not None, apply the assigned transformations
        if self.transform:
            # we don't want to apply transforms like ColorJitter to the mask, so we need to create different transforms
            # for the mask, in this case all we will do is transform it to a tensor
            mask_trans = transforms.Compose([transforms.ToTensor()]) 

            image = self.transform(image)
            mask = mask_trans(mask).long()

        # return the final image and mask
        return image, mask

# Fill in holes in region
""" 
    use the scikit-image region properties to fill holes in regions in the map
    args:
        props: list of RegionProperties objects that were created based on the image
        image: labelled image (used for size)
"""
def FillRegions(props, image):

    new_img = np.zeros((image.shape)).astype('bool') # generate empty boolean array
    mean_area = np.mean([props[i]['filled_area'] for i in range(len(props))])
    mean_solid = np.mean([props[i]['solidity'] for i in range(len(props))])
    
    for i in range(len(props)): # for each region
        # TODO: is there a way I can properly break this out? 
        if props[i]['filled_area'] > 0.3 * mean_area:
            if props[i]['solidity'] > 0.9 * mean_solid:
                (min_row, min_col, max_row, max_col) = props[i]['bbox'] # grab bounding box
                new_img[min_row:max_row, min_col:max_col] = props[i]['filled_image'] # place 'filled_image' array in new_image array at bbox locations

    return new_img # return the newly created image

# postprocessing function for organization
def Postprocess(image):
    #FIXME: something in this is throwing a warning...
    labelled = skiM.label(image.astype('int'), background = 0) # label image
    props = skiM.regionprops(labelled) # generate region props

    filled = FillRegions(props, image) # fill in regions
    return filled

def MaskOverlay(images, masks, ground, batch):

    for i in range(len(images)):
        mask_arr = torch.nn.functional.softmax(masks[i], dim=0).numpy()
        mask_bool = mask_arr[0] < mask_arr[1]

        # SaveImg(Image.fromarray((mask_arr[1]*255).astype('uint8')), "test.png")

        mask_post = Postprocess(mask_bool)

        mask = Image.fromarray(mask_post)
        ground_mask = Image.fromarray(ground[i].squeeze().numpy().astype('bool')).convert(mode="1")

        source = Image.fromarray((images[i].squeeze().numpy()*255).astype('uint8')).convert(mode = "RGBA")

        image = source.copy()

        bg = Image.new(mode="RGBA", size=source.size, color="#00000000")
        
        blue_mask = bg.copy()
        blue = Image.new(mode="RGBA", size=source.size, color="#0000ff80")
        blue_mask.paste(blue, (0,0), mask)

        red_mask = bg.copy()
        red = Image.new(mode="RGBA", size=source.size, color="#ff000080")
        red_mask.paste(red, (0,0), ground_mask)

        image = Image.alpha_composite(image, blue_mask)
        image = Image.alpha_composite(image, red_mask)

        SaveImg(image, f"test_{batch}_{i}.png")

def tensorToImage(img):
    img = np.asarray(img * 255, dtype = np.uint8) # convert to [0..255] uint8 range
    return Image.fromarray(img[0])

## Save Image from PIL image object or torch.Tensor object
"""
quickly save image from tensor or PIL image
args
    img: PIL object or torch.Tensor object
    file_name: file name to be saved to
    location: where on disk to save image
"""
def SaveImg(img, file_name, location = "./data/out"):

    if type(img) == torch.Tensor: # if img is a tensor, we have to convert to save
        img = tensorToImage(img)

    if not os.path.exists(location): os.makedirs(location)
    img.save(os.path.join(location, file_name))

def PredictedVsExpected(pred, exp):

    img_batch = np.zeros((len(pred), 3, pred[0].shape[1], pred[0].shape[2])) 
    for i, (p, e) in enumerate(zip(pred, exp)):
        p = torch.nn.functional.softmax(p, dim=0).numpy()
        p2 = (p[0] < p[1]).astype(int)
        e = e.numpy().astype(int).squeeze(0)

        img_batch[i, 0] = (p2 > e).astype(int) # false positive
        img_batch[i, 2] = (p2 < e).astype(int) # false negative

        p3 = np.where(p2 == 0, -1, p2)
        img_batch[i, 1] = (e == p3).astype(int) # true positive

    return img_batch

