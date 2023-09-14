# ---------------------------------------------------------------------
# Disclaimer: IMPORTANT: This software was developed at the National 
# Institute of Standards and Technology by employees of the Federal 
# Government in the course of their official duties. Pursuant to
# title 17 Section 105 of the United States Code this software is 
# not subject to copyright protection and is in the public domain. 
# This is an experimental system. NIST assumes no responsibility 
# whatsoever for its use by other parties, and makes no guarantees, 
# expressed or implied, about its quality, reliability, or any other 
# characteristic. We would appreciate acknowledgement if the software 
# is used. This software can be redistributed and/or modified freely 
# provided that any derivative works bear some notice that they are 
# derived from it, and any modified versions bear some notice that 
# they have been modified.
# ---------------------------------------------------------------------
from . import cv2, np

class Threshold:

    _defaults = {
        'no_thresh':    False,
        'sigma':        1,
        'boxcar_size':  25,
        'open_radius':  3,
        'thresh':       1e2,
    }

    def __init__(self, img, **kwargs):

        for k, v in Threshold._defaults.items():
            if kwargs.get(k) is not None:
                setattr(self, k, kwargs.get(k))
                continue
            setattr(self, k, v)

        if img.dtype != "float32":
            img = img.astype(np.float32)
        self.img = img

    def _boxcar(self, img):
        
        if img.dtype != "float32":
            img = img.astype(np.float32)
        
        # Low-pass
        kern    = np.ones((2*self.boxcar_size, 2*self.boxcar_size), dtype = np.float32)
        kern    = kern / np.sum(kern)
        img_bkg = cv2.filter2D(img, -1, kern)
        img_sub = img - img_bkg
        img_sub[img_sub < 0] = 0
        return img_sub

    def _gauss(self, img):
        
        if img.dtype != "float32":
            img = img.astype(np.float32)

        # gaussian filter
        mesh     = np.arange(- 4 * self.sigma, 4 * self.sigma + 1, 1)
        x, y     = np.meshgrid(mesh, mesh)
        gauss    = np.exp(-(x * x + y * y) / (2 * self.sigma * self.sigma))
        gauss    = gauss / np.sum(gauss)
        img_filt = cv2.filter2D(img, -1, gauss)
        return img_filt

    def run(self, ):
        # gaussian filter
        img_gauss = self._gauss(self.img)

        # boxcar filter
        img_sub = self._boxcar(img_gauss)

        if self.no_thresh:
            return img_sub

        kern         = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.open_radius, self.open_radius))
        img_bin      = cv2.morphologyEx((img_sub > self.thresh).astype(np.uint8), cv2.MORPH_OPEN, kern, iterations = 3)
        
        return img_bin
