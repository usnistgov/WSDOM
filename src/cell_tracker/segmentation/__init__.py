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

import cv2
import numpy as np 
from ..parallelprocessing   import ParallelProcessor
from ..cell 				import Cell
from .thresholding 			import Threshold
from .pyfogbank				import FogBank
from .segment_fluorescence 	import FluoSegmenter


from ..math_functions import pdist2, matchpairs

from skimage.measure import regionprops

from dataclasses         	import dataclass, field
from typing                 import List

@dataclass
class SegmentationResults:

	true_positives:  int     		= None
	false_positives: int     		= None
	false_negatives: int     		= None
	n_reference:	 int     		= None
	n_test:			 int     		= None
	tp_overlaps: 	 List[float] 	= field(default_factory = list)
	tp_distance: 	 List[float] 	= field(default_factory = list)
	fp_max_overlap:  List[float] 	= field(default_factory = list)
	fn_max_overlap:  List[float] 	= field(default_factory = list)
	fp_min_distance: List[float] 	= field(default_factory = list)
	fn_min_distance: List[float] 	= field(default_factory = list)

	@classmethod
	def from_matchpairs(cls, M, uR, uC, overlaps = None, dsq = None):
		tp_overlaps     = [None]
		tp_distance     = [None]
		fp_max_overlap  = [None]
		fn_max_overlap  = [None]
		fp_min_distance = [None]
		fn_min_distance = [None]

		if overlaps is not None:
			tp_overlaps    = overlaps[M[:, 0], M[:, 1]].tolist()
			fp_max_overlap = np.max(overlaps[uR, :], axis = 1)
			fn_max_overlap = np.max(overlaps[:, uC], axis = 0)

		if dsq is not None:
			tp_distance     = dsq[M[:, 0], M[:, 1]].tolist()
			fp_min_distance = np.min(dsq[uR, :], axis = 1)
			fn_min_distance = np.min(dsq[:, uC], axis = 0)
			
		return cls(
			M.shape[0],
			uR.shape[0],
			uC.shape[0],
			M.shape[0] + uC.shape[0],
			M.shape[0] + uR.shape[0],
			tp_overlaps,
			tp_distance,
			fp_max_overlap,
			fn_max_overlap,
			fp_min_distance,
			fn_min_distance,
			)

class SegmentationAccuracy:

	_defaults = {
		'min_distance': 15 * 15,
		'min_overlap':  0.2,
	}

	@classmethod
	def DataFrame(img_labeled, frame = None):
		reg = regionprops(img_labeled)
		return pd.DataFrame([Cell.from_regionprops(r, frame = frame) for r in reg])

	@classmethod
	def from_images(cls, img_test, img_ref, frame = None, **kwargs):
		df0  = cls.DataFrame(img_test, frame = frame)
		df1  = cls.DataFrame(img_ref, frame = frame)
		return cls.from_dfs(df0, df1, shape = img_test.shape, **kwargs)

	@classmethod
	def from_dfs(cls, df0, df1, shape = None, **kwargs):
		for k, v in cls._defaults.items():
			if kwargs.get(k) is None:
				kwargs.update(k, v)

		dsq = pdist2(df0[["x", "y"]].values, df1[["x", "y"]].values)
		idx = dsq < kwargs.get('min_distance')

		overlaps = None
		if shape is not None:
			overlaps = cls.get_overlaps_matrix(df0, df1, idx, shape = shape)
			dsq[overlaps < kwargs.get('min_overlap')] = 2 * kwargs.get('min_distance')

		M, uR, uC = matchpairs(dsq, kwargs.get('min_distance'))

		return pd.DataFrame([SegmentationResults.from_matchpairs(M, uR, uC, overlaps = overlaps, dsq = dsq)])

	def get_pixellist(x, y, shape):
		return (x * shape[0] + y)

	def compute_overlap(plist0, plist1):
		return np.sum(np.in1d(plist0, plist1)) / np.unique(np.append(plist0, plist1)).shape[0]


	@classmethod
	def get_overlaps_matrix(cls, df0, df1, idx, shape):

		overlaps = np.zeros((df0.shape[0], df1.shape[0]))

		id1 = np.arange(df1.shape[0], dtype = 'int')
		for ii, (index0, row0) in enumerate(df0.iterrows()):

			for jj in id1[idx[ii, :]]:

				row1 = df1.iloc[jj]

				plist0 = cls.get_pixellist(row0["xcoords"], row0["ycoords"], shape = shape)
				plist1 = cls.get_pixellist(row1["xcoords"], row1["ycoords"], shape = shape)
				overlaps[ii, jj] = cls.compute_overlap(plist0, plist1)

		return overlaps
