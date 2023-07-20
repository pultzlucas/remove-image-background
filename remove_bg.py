import torch
import numpy as np
import cv2
from model import U2NET
from torch.autograd import Variable
from skimage import io, transform
from PIL import Image

def remove_bg_from_image(imagePath, output_path):
    model_dir = 'u2net.pth'
    net = U2NET(3, 1)
    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_dir))
        net.cuda()
    else:
        net.load_state_dict(torch.load(model_dir, map_location='cpu'))

    with open(imagePath, "rb") as image:
        f = image.read()
        img = bytearray(f)

    nparr = np.frombuffer(img, np.uint8)

    if len(nparr) == 0:
        return print('empty image')
    try:
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except:
        return print('empty image')

    image = transform.resize(img, (320, 320), mode='constant')

    tmpImg = np.zeros((image.shape[0], image.shape[1], 3))

    tmpImg[:, :, 0] = (image[:, :, 0]-0.485)/0.229
    tmpImg[:, :, 1] = (image[:, :, 1]-0.456)/0.224
    tmpImg[:, :, 2] = (image[:, :, 2]-0.406)/0.225

    tmpImg = tmpImg.transpose((2, 0, 1))
    tmpImg = np.expand_dims(tmpImg, 0)
    image = torch.from_numpy(tmpImg)

    image = image.type(torch.FloatTensor)
    image = Variable(image)

    d1, d2, d3, d4, d5, d6, d7 = net(image)
    pred = d1[:, 0, :, :]
    ma = torch.max(pred)
    mi = torch.min(pred)
    dn = (pred-mi)/(ma-mi)
    pred = dn

    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()
    im = Image.fromarray(predict_np*255).convert('RGB')
    image = io.imread(imagePath)
    imo = im.resize((image.shape[1], image.shape[0]))
    pb_np = np.array(imo)

    # Make and apply mask
    mask = pb_np[:, :, 0]
    mask = np.expand_dims(mask, axis=2)
    imo = np.concatenate((image, mask), axis=2)
    imo = Image.fromarray(imo, 'RGBA')

    # fill white
    fill_color = (255, 255, 255)  # your new background color
    imo = imo.convert("RGBA")   # it had mode P after DL it from OP

    if imo.mode in ('RGBA', 'LA'):
        background = Image.new(imo.mode[:-1], imo.size, fill_color)
        background.paste(imo, imo.split()[-1])  # omit transparency
        imo = background
    imo.convert("RGB").save(output_path)
