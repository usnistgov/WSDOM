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
from . import ParallelProcessor, Cell, FogBank

import os, sys
import skimage.io 		as skio
import numpy 			as np
import pandas 			as pd
from os.path 			import join
from skimage.measure 	import regionprops

class InferencedSegmenter(ParallelProcessor):

	_defaults = {
		'threshold': 2, 
		'common_model_name': '220908_200k_eroded_w2_v{}',
		'image_folder': 	 '220908-200k-eroded-w2-m3-thresh2',
		'iterations':         [1, 2, 3],
		'fogbank_parms':{
			'min_size': 8,
			'min_object_size': 80,
			'erode_size': 2,
			'num_levels': 25,
		},
		'dx': 100,
	}

	def set_common_model_name(self, common_model_name, iterations = None):
		self.common_model_name = common_model_name
		if iterations is not None:
			self.iterations = iterations
		self.models = [self.common_model_name.format(ii) for ii in self.iterations]

	def __init__(self, **kwargs):
		super().__init__(**kwargs)

		for k, v in InferencedSegmenter._defaults.items():
			if kwargs.get(k) is not None:
				setattr(self, k, kwargs.get(k))
				continue
			setattr(self, k, v)
		self.models = [self.common_model_name.format(ii) for ii in self.iterations]

		print(self.models)

	def crop(self, img):
		dx  = self.dx
		img[0:dx, :] = 0 
		img[:, 0:dx] = 0
		img[-dx:, :] = 0
		img[:, -dx:] = 0
		return img
	
	def segment_inferenced(self, root, redo = False, frames = None):

		# Segment fluorescence
		self.load_dir = join(self.root, 'phase-inferenced')
		self.save_dir = join(self.root, self.image_folder)
		os.makedirs(self.save_dir, exist_ok = True)


		filenames     = [filename for filename in os.listdir(join(self.load_dir, self.models[0])) if filename.endswith('.tif')]
		if frames is not None:
			filenames = [filename for filename in filenames if int(filename.split('-')[1][0:-4]) in frames]
		self.redo     = redo

		self.run_series(self.segment, 			filenames, cpus = 20, )


	def segment(self, filename):
		if os.path.isfile(join(self.save_dir, filename)) and not self.redo:
			return
		print(filename)
		img = None
		for folder in self.models:
			if img is None:
				img = skio.imread(join(self.load_dir, folder, filename))
				continue
			img = img + skio.imread(join(self.load_dir, folder, filename))

		img      = self.crop((img >= self.threshold).astype(np.uint8))
		img_seg  = FogBank(img, **self.fogbank_parms).run()

		# Save the image
		skio.imsave(join(self.save_dir, filename), img_seg, check_contrast = False)
