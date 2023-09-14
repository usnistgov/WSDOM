
import skimage.io as skio
import numpy  as np 
import pandas as pd
import sys, os, cv2

from . import pdist2, ParallelProcessor
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
class CellTrackerAnalysis(ParallelProcessor):

	unique = lambda x: x.unique()[0]
	
	def __init__(self, load_path = None, df = None, **kwargs):
		super().__init__(df = df, **kwargs)

	@classmethod
	def get_mother_cells(cls, df = None):
		return df.loc[df["mother_label"].isnull() == False, "mother_label"].unique()

	@classmethod
	def get_daughter_cells(cls, df = None):
		return df.loc[df["mother_label"].isnull() == False, "track_label"]

	@classmethod
	def get_interdivision_cells(cls, df = None):
		mother_cells         = cls.get_mother_cells(df = df)
		inter_division_cells = df.loc[df["track_label"].isin(mother_cells) & 
									 (df["track_label"] != df["lineage_label"]), "track_label"].unique()
		return inter_division_cells.astype('int')

	@classmethod
	def get_interdivision_times(cls, df = None):
		track_start  = np.min
		track_length = lambda x: x.unique().shape[0]

		inter_division_cells  	= cls.get_interdivision_cells(df = df)
		dfc              		= df[df["track_label"].isin(inter_division_cells)].copy()
		inter_division_times  	= pd.DataFrame(
				dfc.pivot_table(index = "track_label", values = "frame", aggfunc = [track_start, track_length])
		)
		columns = inter_division_times.columns
		inter_division_times["interdivision_times"] = inter_division_times[("<lambda>", "frame")]
		inter_division_times["cell_birth"]          = inter_division_times[("amin", "frame")]
		return inter_division_times.drop(columns = columns)
	
	@classmethod
	def get_track_lengths(cls, df = None):
		return df.pivot_table(index = "track_label", values = "frame", aggfunc = lambda x: x.unique().shape[0])

	@classmethod
	def get_divided_cells(cls, df = None):
		return df[df["track_label"] != df["lineage_label"]]["lineage_label"].unique()

	@classmethod
	def get_lineage_lengths(cls, df = None):
		lengths 			    = df.pivot_table(index = "track_label", values = ["frame", "lineage_label"], aggfunc = {"frame":np.max, "lineage_label": cls.unique});
		lineage_starts 		    = df.pivot_table(index = "lineage_label", values = "frame", aggfunc = np.min)
		lengths["track_start"] 	= lineage_starts.loc[lengths["lineage_label"].values, "frame"].values
		lengths 				= lengths[lengths.index.isin(df["mother_label"].unique()) == False]
		lengths["track_length"] = lengths["frame"] - lengths["track_start"]
		lengths["track_end"]   	= lengths["frame"]

		return lengths.drop(columns = "frame")[["lineage_label", "track_start", "track_end", "track_length"]]

	@classmethod
	def get_track_ends(cls, df = None):
		ptable = df.pivot_table(index = "track_label", values = ["frame"], aggfunc = np.max)
		return ptable.merge(df, on = ["track_label", "frame"], how = "left")

	@classmethod
	def get_track_starts(cls, df = None):
		ptable = df.pivot_table(index = "track_label", values = ["frame"], aggfunc = np.min)
		return ptable.merge(df, on = ["track_label", "frame"], how = "left")

	@classmethod
	def remove_short_tracks(cls, df = None, min_length = 50):
		lengths = cls.get_lineage_lengths(df = df)
		return df[df["lineage_label"].isin(lengths[lengths["track_length"] > min_length]["lineage_label"])]

	@classmethod
	def get_msd(cls, df = None, dt = 30):
		column 	= f"MSD-dt{dt:03}"
		lengths = df.pivot_table(index = "track_label", aggfunc = "size")
		
		df0 			= df[df["track_label"].isin(lengths[lenghts > dt].index)].copy()
		df1 			= df0.copy()
		df1["frame"] 	= df1["frame"] - dt 
		df1          	= df1[df1["frame"] > 0].copy()
		df0             = df0.merge(df1, on = ["track_label", "frame"], how = "left", suffixes = ["0", "1"])
		df0["column"] 	= np.square(df0["x1"] - df0["x0"]) + np.square(df0["y1"] - df0["y0"])
		df 				= df.merge(df0[["track_label", "frame", column]], on = ["track_label", "frame"], how = "left")
		return df

	def get_colony_mask(self, frame, load_path = None):
		if not os.path.isfile(os.path.join(self.root, 'phase', f'image-{frame:04}.tif')):
			return None
		try:
			img = skio.imread(os.path.join(self.root, 'colony_labels', f'image-{frame:04}.tif'))
		except FileNotFoundError:
			os.makedirs(os.path.join(self.root, 'colony_labels'), exist_ok = True)
			try:
				cl           = CellList.from_labeled_image(join(self.root, 'phase_inferenced_processed', self.model_name))
				img_colonies = cl.make_image()

				colonies = np.unique(img_colonies)
				colonies = colonies[colonies > 0]

				img = np.zeros(img_colonies.shape, dtype = np.uint16)
				for colony in colonies:
					img[CellList.close_colony_mask(img_colonies, colony_num = colony) > 0] = colony
				skio.imsave(os.path.join(self.root, 'colony_labels', f'image-{frame:04}.tif'), img, check_contrast = False)
			except (TypeError, AttributeError):
				print("file not found")
				return None
		return img

	def get_distance_from_edge(self, frame, load_path = None):
		print(f"frame {frame:04}")

		dft      = self.df[self.df["frame"] == frame].copy()
		img 		= self.get_colony_mask(frame = frame, load_path = load_path)

		if img is None:
			return dft
		
		x = "x"
		if x not in self.df:
			x = "xm"

		y = "y"
		if y not in self.df:
			y = "ym"

		dft["colony_label"] = img[dft[x].values.astype('int'), dft[y].values.astype('int')]
		
		contours = {colony:[] for colony in np.unique(img) if colony > 0}
		for colony in contours.keys():
			tmp, _ = cv2.findContours((img == colony).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
			contours[colony] = tmp

		for index, row in dft.iterrows():
			try:
				dft.loc[index, "distance_from_boundary"] = cv2.pointPolygonTest(contours[row["colony_label"]][0], (row[y], row[x]), True)
				for ii in range(len(contours[row["colony_label"]])):
					if dft.loc[index, "distance_from_boundary"] > 0:
						break
					dft.loc[index, "distance_from_boundary"] = cv2.pointPolygonTest(contours[row["colony_label"]][ii], (row[y], row[x]), True)
			except KeyError:
				pass

		return dft
