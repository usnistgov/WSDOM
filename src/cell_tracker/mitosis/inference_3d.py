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
import sys, os, time, warnings
import tensorflow   as tf 
import numpy        as np
import skimage.io   as skio

import unet_model3

def zscore_normalize(img):
    if len(img.shape) == 2:
        img = img.reshape((1,) + img.shape)

    if not img.dtype == "float32":
        img = img.astype(np.float32)

    for c in range(img.shape[0]):
        std = np.std(img[c, :])
        mn  = np.mean(img[c, :])

        img[c, :] = img[c, :] - mn
        if std > 0:
            img = img / std 
    return img

def normalize(img):
    img                 = zscore_normalize(img)
    rangeMin, rangeMax  = np.max([np.min(img), -5]), np.min([np.max(img),  5])

    img[img>rangeMax] = rangeMax
    img[img<rangeMin] = rangeMin

    pmin, pmax = np.min(img), np.max(img)
    return (((img - pmin) / (pmax - pmin)) * 30000.0).astype(np.float32)


def _inference(img, model):
    return tf.cast(tf.argmax(model(img), axis = -1), tf.uint8)

class Inference3D:

    _defaults = {
        'size_t':   16,
        'size_xy':  128,
        'dt':       4,
        'dxy':      32,
    }

    def __init__(self, root, model_dir, model_name = 'output-well3_4', **kwargs):

        self.load_dir  = os.path.join(root, "phase")
        self.save_dir  = os.path.join(root, model_name)
        self.model_dir = model_dir
        os.makedirs(self.save_dir, exist_ok = True)

        for k, v in Inference3D._defaults.items():
            if kwargs.get(k) is not None:
                setattr(self, k, kwargs.get(k))
                continue
            setattr(self, k, v)

    def load_images(self, gpu_num, t0, t1):
        img_out = None
        for cntr, tt in enumerate(range(t0, t1)):
            filename = f"image-{tt:04}.tif"
            try:
                img = normalize(skio.imread(os.path.join(self.load_dir, filename)))
            except FileNotFoundError:
                continue

            if img_out is None:
                img_out = np.zeros((len(range(t0, t1)), ) + img.shape[1:], dtype = np.float32)
            img_out[cntr, :] = img
        return img_out

    def inf_batch(self, img_stack, model, batch_size = 8):
        
        num_frames, Nx, Ny = img_stack.shape

        Nx_pad = Nx
        if np.mod(Nx, self.size_xy):
            Nx_pad = (1 + Nx//size_xy) * self.size_xy
        Ny_pad = Ny
        if np.mod(Ny, self.size_xy):
            Ny_pad = (1 + Ny//size_xy) * self.size_xy

        Nx_exc = np.mod(Nx_pad, Nx)
        Ny_exc = np.mod(Ny_pad, Ny)

        img_stack_padded = np.zeros((num_frames, Nx_pad, Ny_pad), dtype = np.float32)
        img_stack_padded[:, Nx_exc//2:Nx_exc//2 + Nx, Ny_exc//2:Ny_exc//2 + Ny] = img_stack
        img_inf = np.zeros(img_stack_padded.shape, dtype = np.uint8)
        
        # Create the batch images here that will all go to inference at the same time
        img_batch = np.zeros((batch_size, 1, size_t, size_xy, size_xy), dtype = np.float32)
        txy_list  = np.zeros((batch_size, 3), dtype = 'int')
        
        cntr        = 0
        t0 = time.time()
        for tt in range(0, num_frames, self.size_t - 2 * self.dt):
            print(f"Inferencing frame {tt+dt:04} to {tt+self.size_t-self.dt}")
            if tt + self.size_t > num_frames:
                continue

            for ii in range(0, Nx_pad, self.size_xy - 2 * self.dxy):
                if ii + self.size_xy > Nx_pad:
                    continue
                for jj in range(0, Ny_pad, self.size_xy - 2 * self.dxy):
                    if jj + self.size_xy > Ny_pad:
                        continue

                    img_batch[cntr, 0, :] = img_stack_padded[tt:tt+self.size_t, ii:ii+self.size_xy, jj:jj+self.size_xy]
                    txy_list[cntr, :]     = [tt, ii, jj]
                    cntr = cntr + 1 
                    if cntr == batch_size:
                        img_batch_inf = _inference(img_batch, model)
                        
                        for kk in range(cntr):
                            txy = txy_list[kk, :]
                            tmp = img_inf[txy[0]+self.dt:txy[0]+self.size_t-self.dt, 
                                    txy[1]+self.dxy:txy[1]+self.size_xy-self.dxy,
                                    txy[2]+self.dxy:txy[2]+self.size_xy-self.dxy] 
                            tmp_batch = img_batch_inf[kk, self.dt:self.size_t-self.dt, self.dxy:self.size_xy-self.dxy, self.dxy:self.size_xy-self.dxy]
                            tmp[tmp <= 1] = tmp_batch[tmp <= 1]

                            img_inf[txy[0]+self.dt:txy[0]+self.size_t-self.dt, 
                                    txy[1]+self.dxy:txy[1]+self.size_xy-self.dxy,
                                    txy[2]+self.dxy:txy[2]+self.size_xy-self.dxy] = tmp

                        img_batch = np.zeros((batch_size, 1, self.size_t, self.size_xy, self.size_xy), dtype = np.float32)
                        txy_list  = np.zeros((batch_size, 3), dtype = 'int')
                        cntr = 0

        # Before you return, make sure the batch is empty
        for kk in range(cntr):
            txy = txy_list[kk, :]
            img_inf[txy[0]+self.dt:txy[0]+self.size_t-self.dt, 
                    txy[1]+self.dxy:txy[1]+self.size_xy-self.dxy,
                    txy[2]+self.dxy:txy[2]+self.size_xy-self.dxy] = \
                        img_batch_inf[kk, self.dt:self.size_t-self.dt, self.dxy:self.size_xy-self.dxy, self.dxy:self.size_xy-self.dxy]

        return img_inf[:, Nx_exc//2:Nx_exc//2 + Nx, Ny_exc//2:Ny_exc//2 + Ny]  

    def inference_chunk(self, t0 = 0, t1 = 16):
        gpu_name = "GPU:0"
        
        img = self.load_images(gpu_num, t0, t1)
        gpu_devices = tf.config.experimental.list_physical_devices('GPU')
        for dev in gpu_devices:
            tf.config.experimental.set_memory_growth(dev, True)

        # Read in the images 
        with tf.device(gpu_name):
            model   = tf.saved_model.load(os.path.join(self.model_dir, self.model_name))
            img_inf = self.inf_batch(img, model)

        if t0 == 0:
            for cntr, tt in enumerate(range(t0, t1-self.dt)):
                skio.imsave(os.path.join(self.save_dir, f"image-{tt:04}.tif"), img_inf[cntr, :], check_contrast = False)
            return
        for cntr, tt in enumerate(range(t0+self.dt, t1-self.dt)):
            skio.imsave(os.path.join(self.save_dir, f"image-{tt:04}.tif"), img_inf[cntr+4, :], check_contrast = False)

if __name__ == "__main__":
    date        = "230427"
    wells       = [3, 4, 5]
    root        = f'/mnt/x/InScoper/{date}_H2B_Live'
    model_name  = "output-well3_4"
    subfolder   = "stitched_images"
    gpu_num = 0 
    if len(sys.argv) > 1:
        gpu_num = int(sys.argv[1])
    for w in wells:
        print(f"Inferencing well {w} from experiment {date}") 
        well = f"well{w:02}"
        model_dir = "/home/zab1/data/mitosis_models"
        mit = MitosisInferencer(root = os.path.join(root, well), subfolder = subfolder, model_dir = model_dir, model_name = model_name)
        mit.inference_chunk(gpu_num)
