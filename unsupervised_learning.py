import cv2
import numpy as np
import secrets

import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from PIL import Image
import os
import numpy as np
import sys

from models.vgg19 import VGGUNET19

def unsupervised_learning(model, imagename):
    
    