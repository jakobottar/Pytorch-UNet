import dataIO

def parseTxt(file):
    f = open(file, "r")
    return [X.rstrip() for X in f.readlines()]

# get image file locations from text file
# TODO: condense both of these into a json
train_images = parseTxt("data/CALC-images.txt")
train_masks = parseTxt("data/CALC-masks.txt")

# feed into preprocessor
dataIO.Preprocess(train_images, train_masks, 30, 440, 440, 4)
# logger.info("finished preprocessing raw images.")