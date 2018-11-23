import tesserocr
from PIL import Image
# Morphological filtering
from skimage.morphology import opening
from skimage.morphology import disk

# Data handling
import numpy as np

# Connected component filtering
# import cv2
# print tesserocr.tesseract_version()  # print tesseract-ocr version
# print tesserocr.get_languages()  # prints tessdata path and list of available languages
#
# image = Image.open('/Users/mayuukh/Documents/tessaract/sample.jpg')
# print tesserocr.image_to_text(image)  # print ocr text from image
# # or
# print tesserocr.file_to_text(image)

with tesserocr.PyTessBaseAPI() as api:
    image = Image.open('/Users/mayuukh/Documents/tessaract/sample2.jpg')
    #
    # black = (0, 0, 0)
    # white = (255, 255, 255)
    # threshold = (160, 160, 160)
    #
    # # Open input image in grayscale mode and get its pixels.
    # img = image.convert("LA")
    # # pixels = img.getdata()
    # #
    # # newPixels = []
    # #
    # # # Compare each pixel
    # # for pixel in pixels:
    # #     if pixel < threshold:
    # #         newPixels.append(black)
    # #     else:
    # #         newPixels.append(white)
    # #
    # # # Create and save new image.
    # # newImg = Image.new("RGB", img.size)
    # # newImg.putdata(newPixels)
    # # newImg.save("newImage.jpg")
    # pixels = np.array(img)[:, :, 0]
    #
    # # Remove pixels above threshold
    # pixels[pixels > threshold] = white
    # pixels[pixels < threshold] = black
    #
    # # Morphological opening
    # blobSize = 1  # Select the maximum radius of the blobs you would like to remove
    # structureElement = disk(blobSize)  # you can define different shapes, here we take a disk shape
    # # We need to invert the image such that black is background and white foreground to perform the opening
    # pixels = np.invert(opening(np.invert(pixels), structureElement))
    #
    # # Create and save new image.
    # newImg = Image.fromarray(pixels).convert('RGB')
    # newImg.save("newImage1.PNG")
    #
    # # Find the connected components (black objects in your image)
    # # Because the function searches for white connected components on a black background, we need to invert the image
    # nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(np.invert(pixels), connectivity=8)
    #
    # # For every connected component in your image, you can obtain the number of pixels from the stats variable in the last
    # # column. We remove the first entry from sizes, because this is the entry of the background connected component
    # sizes = stats[1:, -1]
    # nb_components -= 1
    #
    # # Define the minimum size (number of pixels) a component should consist of
    # minimum_size = 100
    #
    # # Create a new image
    # newPixels = np.ones(pixels.shape) * 255
    #
    # # Iterate over all components in the image, only keep the components larger than minimum size
    # for i in range(1, nb_components):
    #     if sizes[i] > minimum_size:
    #         newPixels[output == i + 1] = 0
    #
    # # Create and save new image
    # newImg = Image.fromarray(newPixels).convert('RGB')
    # newImg.save("newImage2.PNG")
    api.SetImage(image)
    boxes = api.GetComponentImages(tesserocr.RIL.TEXTLINE, True)
    print 'Found {} textline image components.'.format(len(boxes))
    for i, (im, box, _, _) in enumerate(boxes):
        # im is a PIL image object
        # box is a dict with x, y, w and h keys
        api.SetRectangle(box['x'], box['y'], box['w'], box['h'])
        ocrResult = api.GetUTF8Text()
        conf = api.MeanTextConf()
        print (u"Box[{0}]: x={x}, y={y}, w={w}, h={h}, "
               "confidence: {1}, text: {2}").format(i, conf, ocrResult, **box)
