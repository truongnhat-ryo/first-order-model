from facenet_pytorch import MTCNN
import torch
import numpy as np
import  cv2
from PIL import Image, ImageDraw
#from IPython import display
import os
import PIL
import torch.nn.functional as F
import math
from torchvision import transforms
import numpy as np
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))
mtcnn = MTCNN(keep_all=True, device=device)

def face_detetect(img, scale=1.2):
    '''
    --input: pil image
    --output: pil image
    '''
    boxes, _ = mtcnn.detect(img)
    box = boxes[0]
    l,t,r,b = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    width, height = img.size
    scale_l = scale - 1 + 0.1
    scale_r = scale - 1 + 0.1
    scale_t = scale - 1 + 0.1
    scale_b = scale - 1 - 0.1
    w, h = r-l, b-t
    if(t - scale_t*h < 0):
        t = 1
    else:
        t -= int(scale_t*h)
    if(b + scale_b*h > height):
        b = height
    else:
        b += int(scale_b*h)
    if(l - scale_l*w > 0):
        l -= int(scale_l*w)
    else:
        l=1
    if(r + scale_r*w < width):
        r += int(scale_r*w)
    else:
        r = width
    w1, h1 = r-l, b-t
    img_crop = img.crop((l,t,r,b))
    quad = np.array([[l,t],[l,b],[r,b],[r,t]])
    img_result = img.transform((256,256),PIL.Image.QUAD, (quad + 0.5).flatten(),
                            PIL.Image.BILINEAR)
    return img_result,l,t,r,b
