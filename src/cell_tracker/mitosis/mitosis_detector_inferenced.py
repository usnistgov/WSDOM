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
import cv2, sys, os
import numpy  		as np 
import pandas 		as pd
import skimage.io 	as skio

from os.path 			import join
from skimage.measure 	import regionprops, label 
from dataclasses 		import dataclass
from ..parallelprocessing import ParallelProcessor
@dataclass
class Mitosis:

	x: 			float = None
	y: 			float = None
	area: 		int   = None
	start: 		int   = None
	end: 		int   = None

	x0: int = None
	x1: int = None
	y0: int = None
	y1: int = None

	xm:  float = None
	ym:  float = None
	xd1: float = None
	yd1: float = None
	xd2: float = None
	yd2: float = None
	frame: int = None

	@classmethod
	def from_regionprops(cls, reg):
		x       = reg.centroid[0]
		y       = reg.centroid[1]
		area 	= reg.area
		end     = reg.bbox[-1]
		start   = reg.bbox[2]
		bbox 	= reg.bbox

		x0, y0, x1, y1 = bbox[0], bbox[1], bbox[3], bbox[4]

		return cls(x = x, y = y, area = area, start = start, end = end, x0 = x0, y0 = y0, x1 = x1, y1 = y1)

class MitosisDetector(ParallelProcessor):

	_defaults   = {
		'root': 					'Z:/Analysis/InScoper/230202_H2B_Live/well03/stitched_images',
		'model_name': 				'output-well3_4',
		'min_size': 				50,
		'dilation_size': 			3,
		'daughter_size_similarity': 0.2,
		'min_daughter_size': 		50,
		'min_mother_size':   		100,
		'min_volume': 				2e2,
		'min_duration': 			7,
	}

	_default_save_filename = 'xyt_divisions_from_U-Net.pkl'

	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		for k, v in MitosisDetector._defaults.items():
			if kwargs.get(k) is None:
				setattr(self, k, v)
				continue
			setattr(self, k, kwargs.get(k))


		if not hasattr(self, 'mitosis_filename'):
			self.mitosis_filename = self._default_save_filename

	# preprocess is just a dilation
	def preprocess(self, frame, ):
		kern                    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.dilation_size, self.dilation_size))
		img 					= skio.imread(join(self.root, self.model_name, f'image-{frame:04}.tif'))
		img                     = cv2.dilate((img > 1).astype(np.uint8), kern, iterations = 2)
		[num_labels, cc, stats] = cv2.connectedComponentsWithStats(img)[0:3]
		labels_to_remove        = [label for label in range(1, num_labels) if stats[label, cv2.CC_STAT_AREA] < self.min_size]
		cc[np.in1d(cc.flatten(), labels_to_remove).reshape(img.shape)] = 0
		return (cc > 0).astype(np.uint8)

	# Finds all 3D blobs that are either class 2 or 3 
	def _load(self, t0, tf):
		images = None
		for ii, frame in enumerate(range(t0, tf)):
			img =  self.preprocess(frame = frame)
			if images is None:
				images = np.zeros(img.shape + (tf - t0, ), dtype = np.uint8)
			images[:, :, ii] = img
		reg = regionprops(label(images > 0))
		df  = pd.DataFrame([Mitosis.from_regionprops(r) for r in reg])
		return df

	# Loads in the entire three-dimensional z-stack
	def load_images(self, t0, tf):
		images = None
		for ii, frame in enumerate(range(t0, tf)):
			img = skio.imread(join(self.root, self.model_name, f'image-{frame:04}.tif'))
			if images is None:
				images = np.zeros(img.shape + (tf - t0, ), dtype = np.uint8)
			images[:, :, ii] = img
		return images

	def find_xyt(self, images, row):

		t0, tf = int(row["start"]), int(row["end"])
		x0, x1 = int(row["x0"]), int(row["x1"])
		y0, y1 = int(row["y0"]), int(row["y1"])

		tf     = np.min([tf + 3, images.shape[-1]-1])
		imgs   = images[x0:x1, y0:y1, t0:tf]

		kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
		for ii in range(imgs.shape[-1]):

			img_daughters = label(cv2.dilate((imgs[:, :, ii] == 3).astype(np.uint8), kern, iterations = 1) > 0)
			reg           = regionprops(img_daughters)
			if len(reg) != 2:
				continue

			areas = np.array([r.area for r in reg])
			if( np.diff(areas) / np.sum(areas) > self.daughter_size_similarity) or np.any(areas < self.min_daughter_size): 
				continue

			row["frame"] = t0 + ii - 1
			row["xd1"], row["yd1"]   = reg[0].centroid[0] + x0, reg[0].centroid[1] + y0
			row["xd2"], row["yd2"]   = reg[1].centroid[0] + x0, reg[1].centroid[1] + y0

			# Get the mother
			img = images[x0:x1, y0:y1, int(row["frame"])] > 1
			reg = regionprops(label(img))

			if len(reg) > 1:
				max_areas = np.max([r.area for r in reg])
				reg       = [r for r in reg if r.area == max_areas]

			if len(reg) == 0:
				continue

			row["xm"], row["ym"] = reg[0].centroid[0] + x0, reg[0].centroid[1] + y0
			
			return row

		max_area, max_ii, max_reg = None, None, None
		for ii in range(imgs.shape[-1]):

			img_labeled = label(imgs[:, :, ii] > 1)
			reg 		= regionprops(img_labeled)

			if len(reg) == 0:
				continue

			if len(reg) > 1:
				ma  = np.max([r.area for r in reg])
				reg = [r for r in reg if r.area == ma]

			if max_area is None:
				continue

			if (max_area is None) or (reg[0].area > max_area):
				max_area, max_ii, max_reg = reg[0].area, ii, reg
				continue

		if max_area is None:
			return row

		if max_area < self.min_mother_area:
			return row

		if max_ii < ii:
			orientation = max_reg[0].orientation
			major       = max_reg[0].axis_major_length

			x, y 		= max_reg[0].centroid[0], max_reg[0].centroid[1]

			row["xm"]    = x + x0
			row["ym"]    = y + y0
			row["frame"] = t0 + max_ii

			row["xd1"]   = x + 0.6 * 0.5*major * np.cos(orientation) + x0
			row["yd1"]   = y + 0.6 * 0.5*major * np.sin(orientation) + y0
			row["xd2"]   = x - 0.6 * 0.5*major * np.cos(orientation) + x0
			row["yd2"]   = y - 0.6 * 0.5*major * np.sin(orientation) + y0

		return row

	def run(self, filename = None, **kwargs):

		try:
			raise FileNotFoundError
			df = pd.read_pickle(join(self.root, self.mitosis_filename))
			return
		except FileNotFoundError:
			pass

		# Get the frames of interest
		t0, tf  = 0, len(os.listdir(join(self.root, self.model_name)))
		print((t0, tf))

		if kwargs.get('t0') is not None:
			t0 = kwargs.get('t0')
		if (kwargs.get('tf') is not None):
			tf = np.min([tf, kwargs.get('tf')])

		print((t0, tf))
		try:
			raise FileNotFoundError
			df = pd.read_pickle(join(self.root, 'xyt_divisions_mother_cells_only.pkl'))
		except FileNotFoundError:
			df      = self._load(t0, tf)
			df.to_pickle(join(self.root, 'xyt_divisions_mother_cells_only.pkl'))

		df      = df[df["end"] - df["start"] > self.min_duration]
		df      = df[df["area"] > self.min_volume]

		print(f"found {df.shape[0]} potential mitotic events")

		print("loading 3D U-Net")
		images  = self.load_images(t0, tf)

		print("getting mitosis xyt")
		for index, row in df.iterrows():
			new_row 		= self.find_xyt(images, row)
			df.loc[index] 	= new_row

		df = df.dropna()
		df = df.sort_values(by = "frame")
		df.to_pickle(join(self.root, self.mitosis_filename))
		print(f"found {df.shape[0]} mitoses")
