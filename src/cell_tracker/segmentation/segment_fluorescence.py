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
from . import ParallelProcessor, Cell, FogBank, Threshold

import os
import skimage.io 		as skio
import numpy 			as np
import pandas 			as pd
from os.path 			import join
from skimage.measure 	import regionprops

class FluoSegmenter(ParallelProcessor):

	_defaults = {
		'thresh_parms': {
			'sigma': 		1,
			'boxcar_size': 	25,
			'thresh':      1e2,

		},
		'fogbank_parms':{
			'min_size': 8,
			'min_object_size': 120,
			'erode_size': 4,
			'num_levels': 25,
		},

		'mitosis_parms': {
			'no_thresh': True, 
			'sigma': 1,
			'boxcar_size': 15, 
			'thresh': 8e2,
		},
		'dx': 100,
	}

	def __init__(self, **kwargs):
		super().__init__(**kwargs)

		for k, v in FluoSegmenter._defaults.items():
			if kwargs.get(k) is not None:
				setattr(self, k, kwargs.get(k))
				continue
			setattr(self, k, v)

	def normalize(self, img):
		img = img.astype(np.float32)
		img = (img - np.mean(img)) / np.std(img)
		img[img < -5] = -5
		img[img >  5] =  5
		return (255 * (img - np.min(img)) / (np.max(img) - np.min(img))).astype(np.uint8)

	def crop(self, img):
		dx  = self.dx
		img[0:dx, :] = 0 
		img[:, 0:dx] = 0
		img[-dx:, :] = 0
		img[:, -dx:] = 0
		return img
	
	def segment_fluorescence(self, load_dir, do_mitosis = True, redo = False, frames = None):

		filenames     = [filename for filename in os.listdir(load_dir) if filename.endswith('.tif')]
		if frames is not None:
			filenames = [filename for filename in filenames if int(filename.split('-')[1][0:-4]) in frames]
		self.redo     = redo

		# Segment fluorescence
		self.load_dir = load_dir
		self.save_dir = f'{load_dir}-segmented'
		os.makedirs(self.save_dir, exist_ok = True)
		self.run_series(self.segment, 			filenames, cpus = 20, )

		if not do_mitosis:
			return
		
		# Segment mitosis from fluorescence
		self.fluo_dir = self.load_dir
		self.load_dir = self.save_dir
		self.save_dir = f'{self.save_dir}-mitosis'
		os.makedirs(self.save_dir, exist_ok = True)
		self.run_series(self.segment_mitosis, 	filenames, cpus = 20, )

	def segment(self, filename):
		if os.path.isfile(join(self.save_dir, filename)) and not self.redo:
			return
		print(filename)
		img = skio.imread(join(self.load_dir, filename))
		
		# Thresholding
		img_thresh = Threshold(img, **self.thresh_parms).run()
		img_thresh = self.crop(img_thresh)

		# Segmentation
		img = self.normalize(img)
		img[img_thresh == 0] = 0

		img_seg    = FogBank(img_thresh, **self.fogbank_parms).run(img_to_transform = img)

		# Save the image
		skio.imsave(join(self.save_dir, filename), img_seg, check_contrast = False)

	def segment_mitosis(self, filename = None):
		if os.path.isfile(join(self.save_dir, f'df-{filename[0:-4]}.pkl')):
			return

		img     = skio.imread(join(self.fluo_dir, filename))
		img_seg = skio.imread(join(self.load_dir, filename))
		
		img_thresh = Threshold(img, **self.mitosis_parms).run()
		
		reg        = regionprops(img_seg, intensity_image = img_thresh)
		reg        = [r for r in reg if r.intensity_mean > self.mitosis_parms.get('thresh')]

		df      = pd.DataFrame([Cell.from_regionprops(r, frame = int(filename.split('-')[1][0:-4])) for r in reg])
		df.to_pickle(join(self.save_dir, f'df-{filename[0:-4]}.pkl'))

