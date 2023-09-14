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
import os

from os.path import join
from . import matchpairs, pdist2
from dataclasses import dataclass

import pandas as pd
import numpy  as np
@dataclass
class Mitosis:

	frame: int = None
	xm:    int = None
	ym:    int = None
	xd1:   int = None
	yd1:   int = None
	xd2:   int = None
	yd2:   int = None

	@classmethod
	def from_dfs(cls, dfm, dfd):
		frame 		= dfm["frame"].values[0]
		xm 			= dfm["x"].values[0]
		ym 		 	= dfm["y"].values[0]
		xd1, xd2 	= dfd["x"].values.tolist()
		yd1, yd2 	= dfd["y"].values.tolist()
		return cls(frame = frame, xm = xm, ym = ym, xd1 = xd1, xd2 = xd2, yd1 = yd1, yd2 = yd2)

class MitosisTrackerFluo:
	_defaults = {
		'filename': 'df-image-{:04d}.pkl',
		'dt':       1,
		'distance_cutoff': 25 * 25,
		'size_similarity': 0.2,
		'polarity':        -0.45,
		'mitosis_filename': 'xyt_divisions_from_fluorescence.csv',
	}
	def __init__(self, root = None, load_dir = None, **kwargs):

		for k, v in MitosisTrackerFluo._defaults.items():
			if kwargs.get(k) is not None:
				setattr(self, k, kwargs.get(k))
				continue
			setattr(self, k, v)

		self.root     = root
		self.load_dir = load_dir

	def run(self, ):
		mitosis = pd.DataFrame()

		for frame in range(0, 100000, self.dt):
			if not os.path.isfile(join(self.load_dir, self.filename.format(frame+1))):
				break

			if not frame:
				df0 = pd.read_pickle(join(self.load_dir, self.filename.format(frame)))

			df1     = pd.read_pickle(join(self.load_dir, self.filename.format(frame + self.dt)))
			df0_dup = pd.concat([df0, df0[df0["mitosis"] == False]], ignore_index = True)

			dsq       = pdist2(df0_dup[["x", "y"]].values, df1[["x", "y"]].values)
			M, uR, uC = matchpairs(dsq, D_cutoff = self.distance_cutoff)

			pairs 			= pd.DataFrame(df0_dup.loc[M[:, 0], "label"]).reset_index(drop = True)
			pairs["label1"] = pd.DataFrame(df1.loc[M[:, 1], "label"]).reset_index(drop = True)

			ptable 		= pairs.pivot_table(index = "label", aggfunc = "size")
			duplicates 	= pairs[pairs["label"].isin(ptable[ptable == 2].index)].pivot_table(index = "label", values = "label1", aggfunc = lambda x: [x.unique()])

			for index, row in duplicates.iterrows():
				dfm = df0[df0["label"] == index]
				dfd = df1[df1["label"].isin(row[0][0])]

				# check size similarity
				if np.abs(np.diff(dfd["area"].values)) / np.sum(dfd["area"]) > self.size_similarity:
					continue

				# check opposite vectors
				dx = dfd["x"].values - dfm["x"].values
				dy = dfd["y"].values - dfm["y"].values
				d  = np.sqrt(dx * dx + dy * dy)
				
				dx, dy = dx / d, dy / d

				if np.prod(dx) + np.prod(dy) > self.polarity:
					continue

				df0_dup.loc[df0_dup["label"] == index, "mitosis"] = True
				
				mitosis = pd.concat([mitosis, pd.DataFrame([Mitosis.from_dfs(dfm, dfd)])], ignore_index = True)
			

			df1.loc[M[:, 1], "mitosis"] = df0_dup.loc[M[:, 0], "mitosis"].values
			df0 = df1

		if self.mitosis_filename is None:
			return mitosis

		mitosis.to_csv(join(self.root, self.mitosis_filename))
