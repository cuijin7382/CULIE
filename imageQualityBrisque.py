import matplotlib.pyplot as plt
import imquality.brisque as brisque
import PIL.Image
import cv2
# import  matplotlib.pyplot as plt
path1='./1illu.bmp'

img1=PIL.Image.open(path1)
# print(img1)
plt.imshow(img1)
plt.axis('off')
print("image-quality{}".format(brisque.score(img1)))