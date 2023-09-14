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
import sys, time

from .celltrackerfactory 	import CellTrackerFactory
from . 						import os, np, pd, pdist2, matchpairs, Cell
from scipy 					import interpolate
from skimage.measure 		import regionprops

class CellTracker(CellTrackerFactory):
	columns = ["xcoords", "ycoords", "bbox"]
	MAX_VALUE = np.finfo(np.float64).max

	def __init__(self, root, load_dir, img_shape = None, mitosis_filename = None, **kwargs):
		super().__init__(**kwargs)
		
		self.root        = root
		self.load_dir    = load_dir
		
		self.limits = kwargs.get("limits")
		if self.limits is None:
			self.limits = []
			
		self.dtf_mitosis = pd.DataFrame()
		while True:
			if mitosis_filename is None:
				break

			print(f'Loading mitosis from {mitosis_filename}')
			if mitosis_filename.endswith('pkl'):
				self.dtf_mitosis = pd.read_pickle(os.path.join(self.root, mitosis_filename))
			if mitosis_filename.endswith('csv'):
				self.dtf_mitosis = pd.read_csv(os.path.join(self.root, mitosis_filename))

			if self.dtf_mitosis.shape[0] == 0:
				break

			self.dtf_mitosis["lineage_label"]    = None
			self.dtf_mitosis["track_label"]      = None
			self.dtf_mitosis["daughters_found"]  = False
			self.mit_cutoff                      = 8 * 8

			if len(self.limits) > 0:
				self.dtf_mitosis = self.dtf_mitosis.loc[
													(self.dtf_mitosis["xm"] > self.limits[0]) & 
													(self.dtf_mitosis["xm"] < self.limits[2]) &
													(self.dtf_mitosis["ym"] > self.limits[1]) &
													(self.dtf_mitosis["ym"] < self.limits[3])
												   ]
			break

		if img_shape is not None:
			self.shape = img_shape

	# initialize the first frame
	def _init(self, frame):

		df = pd.read_pickle(os.path.join(self.load_dir, f'cells-{frame:04}.pkl'))

		df["track_label"]   = np.arange(df.shape[0], dtype = 'int')
		df["lineage_label"] = np.arange(df.shape[0], dtype = 'int')
		df["new_cell"]      = True
		df["track_length"]  = 0
		self.max_label      = df.shape[0]    
		
		return df
	
	# MITOSIS MATCH PAIRING IS RIGHT HERE
	def _mitosis(self, xy0, xy1, frame, **kwargs):
		if self.dtf_mitosis.shape[0] == 0:
			return None

		self.mit  = self.dtf_mitosis[(self.dtf_mitosis["frame"] <= frame)]
		self.mit  = self.mit[self.mit["track_label"].isnull() | 
								 (self.mit["daughters_found"] == False)]
		
		# Frame check -> If I cannot find the mitosis, then I will never find it. 
		self.mit  = self.mit[self.mit["frame"] > frame - self.mitosis_time_cutoff]
		if self.mit.shape[0] == 0:
			return None
		
		if len(self.limits) > 0:
			xym     = np.array([[r["xm"]-self.limits[0], r["ym"]-self.limits[1]] for index, r in self.mit.iterrows()])
		else:
			xym     = np.array([[r["xm"], r["ym"]] for index, r in self.mit.iterrows()])
			
		dsq     = pdist2(xy0, xym)
		dsq_min = np.min(dsq, axis = 0)
		
		id0m    = np.zeros(dsq_min.shape)
		for ii in range(len(id0m)):
			id0m[ii] = self.id0[dsq[:, ii] == dsq_min[ii]]

		mothers_to_keep = dsq_min <= self.mit_cutoff
		mothers_to_keep = mothers_to_keep * (self.mit["track_label"].isnull().values) > 0
		
		# Update the mitosis dataframe with the true mothers
		true_mothers  = id0m[mothers_to_keep]
		
		self.mit.loc[mothers_to_keep, "lineage_label"]       						= self.dtf0.loc[true_mothers, "lineage_label"].values
		self.dtf_mitosis.loc[self.mit.loc[mothers_to_keep].index, "lineage_label"] 	= self.dtf0.loc[true_mothers, "lineage_label"].values
		self.mit.loc[mothers_to_keep, "track_label"]         						= self.dtf0.loc[true_mothers, "track_label"].values
		self.dtf_mitosis.loc[self.mit.loc[mothers_to_keep].index, "track_label"] 	= self.dtf0.loc[true_mothers, "track_label"].values
		
		# Daughter one
		if len(self.limits) > 0:
			xyd1    = np.array([[r["xd1"]-self.limits[0], r["yd1"]-self.limits[1]] for index, r in self.mit.iterrows()])
		else:
			xyd1    = np.array([[r["xd1"], r["yd1"]] for index, r in self.mit.iterrows()])
		
		dsq     = pdist2(xy1, xyd1)
		dsq_min = np.min(dsq, axis = 0) 
		id1d1   = np.zeros(dsq_min.shape)
		for ii in range(len(id1d1)):
			id1d1[ii] = self.id1[dsq[:, ii] == dsq_min[ii]]
		
		daughters_to_keep = dsq_min <= self.mit_cutoff
		
		# Daughter two
		if len(self.limits) > 0:
			xyd2    = np.array([[r["xd2"]-self.limits[0], r["yd2"]-self.limits[1]] for index, r in self.mit.iterrows()])
		else:
			xyd2    = np.array([[r["xd2"], r["yd2"]] for index, r in self.mit.iterrows()])
			
		dsq     = pdist2(xy1, xyd2)
		dsq_min = np.min(dsq, axis = 0)
		id1d2   = np.zeros(dsq_min.shape)
		for ii in range(len(id1d2)):
			id1d2[ii] = self.id1[dsq[:, ii] == dsq_min[ii]]
		
		daughters_to_keep = daughters_to_keep * (dsq_min <= self.mit_cutoff) > 0
		
		dfd1   = self.dtf1.loc[id1d1]
		dfd2   = self.dtf1.loc[id1d2]
		
		# Check if they map to the same daughter
		daughters_to_keep = daughters_to_keep * (dfd1.index != dfd2.index)
		daughters_to_keep = daughters_to_keep * (self.mit["lineage_label"].isnull().values == False) > 0
		id1d1 = id1d1[daughters_to_keep]
		id1d2 = id1d2[daughters_to_keep]
		if len(id1d1) == 0:
			return None
		
		self.dtf_mitosis.loc[self.mit.loc[daughters_to_keep].index, "daughters_found"] = True
		self.mit.loc[daughters_to_keep, "daughters_found"] = True
		self.dtf1.loc[id1d1, "lineage_label"] = self.mit.loc[daughters_to_keep, "lineage_label"].values
		self.dtf1.loc[id1d2, "lineage_label"] = self.mit.loc[daughters_to_keep, "lineage_label"].values
		
		self.dtf1.loc[id1d1, "track_label"]   	= self.max_label + np.arange(np.sum(daughters_to_keep), dtype = 'int')
		self.max_label 							= self.max_label + np.sum(daughters_to_keep)
		self.dtf1.loc[id1d2, "track_label"]   	= self.max_label + np.arange(np.sum(daughters_to_keep), dtype = 'int')
		self.max_label 							= self.max_label + np.sum(daughters_to_keep)
		
		self.dtf0.loc[np.in1d(self.dtf0["track_label"].values, self.mit.loc[daughters_to_keep, "track_label"].values), "mitosis"] = True
		self.dtf1.loc[id1d1, "mitosis"] 		= True
		self.dtf1.loc[id1d2, "mitosis"] 		= True
		self.dtf1.loc[id1d1, "mother_label"] 	= self.mit.loc[daughters_to_keep, "track_label"].values
		self.dtf1.loc[id1d2, "mother_label"] 	= self.mit.loc[daughters_to_keep, "track_label"].values
		return None
	
	def _run(self, t0 = 0, tf = 0, save_path = None, dt = 1, frames = None):
		columns = CellTracker.columns
		if frames is None:
			frames = range(t0, tf+1, dt)

		self.dt = dt
		if not os.path.isfile(os.path.join(self.load_dir, f'cells-{frames[-1]+dt:04}.pkl')):
			frames = frames[0:-1]
			while True:
				if not os.path.isfile(os.path.join(self.load_dir, f'cells-{frames[-1]+dt:04}.pkl')):
					frames = frames[0:-1]
					continue
				break

		with open(save_path.replace('csv', 'txt'), 'w') as log_file:
			log_file.write(save_path)
			log_file.write('\n')

		for frame in frames:
			with open(save_path.replace('csv', 'txt'), 'a') as log_file:
				time0 = time.time()
				if frame == t0:
					self.dtf0 = self._init(frame)
					self.dtf0.drop(columns = columns).to_csv(save_path, index = False)
					columns = [column for column in columns if column in self.dtf0]
					log_file.write("Initializing\n")
					log_file.write(f'dropping these columns: {columns}\n')
				# Load frame 1 regionprops

				self.dtf1 = pd.read_pickle(os.path.join(self.load_dir, f'cells-{frame+dt:04}.pkl'))
				self.dtf1["track_length"] = 0
				self.dtf1["new_cell"]     = None

				while True:
					# Index of the dataframe
					self.id0  = self.dtf0.index.values
					self.id1  = self.dtf1.index.values
					
					xy0       = self._centroid(self.dtf0)
					xy1       = self._centroid(self.dtf1)
					
					mit       = self._mitosis(xy0, xy1, frame = frame)
					
					self.id0  = self.id0[self.dtf0["mitosis"] == False]
					self.id1  = self.id1[self.dtf1["mitosis"] == False]
					
					xy0       = xy0[self.dtf0["mitosis"] == False, :]
					xy1       = xy1[self.dtf1["mitosis"] == False, :]

					if mit == 0:
						return
					
					dsq                                      = pdist2(xy0, xy1)
					dsq[dsq > self.max_disp * self.max_disp] = self.MAX_VALUE
					idx                                      = dsq < self.MAX_VALUE
					
					id0 			= np.arange(idx.shape[0], dtype = 'int')
					id1 			= np.arange(idx.shape[1], dtype = 'int')
					deg0          = np.sum(idx, axis = 1)
					self.overlaps = np.zeros(dsq.shape)
					for ii in id0[deg0 > 0]:
						plist0 = self._pixellist(self.dtf0.loc[self.id0[ii], "xcoords"], self.dtf0.loc[self.id0[ii], "ycoords"])
						for jj in id1[idx[ii, :]]:
							plist1                = self._pixellist(self.dtf1.loc[self.id1[jj], "xcoords"], self.dtf1.loc[self.id1[jj], "ycoords"])
							self.overlaps[ii, jj] = np.sum(np.in1d(plist0, plist1)) / len(plist0)
			
			
					dsq[self.overlaps < self.min_overlap] = self.MAX_VALUE
					idx[self.overlaps < self.min_overlap] = 0
					
					cost                        = dsq.copy()
					cost[cost < self.MAX_VALUE] =  1.0 * cost[cost < self.MAX_VALUE] / np.square(self.max_disp)
					cost[cost < self.MAX_VALUE] =  cost[cost < self.MAX_VALUE] + 0.4 * (1 - self.overlaps[cost < self.MAX_VALUE])

					
					# pair together the objects
					self.M, self.uR, self.uC = matchpairs(cost)

					if self.M is None:
						self.uR = np.arange(xy0.shape[0], dtype = 'int')
						self.uC = np.arange(xy1.shape[0], dtype = 'int')
						self.M  = None


					try:
						log_file.write(f'{self.M.shape[0]} matches, {self.uR.shape[0]} unmapped rows, {self.uC.shape[0]} unmapped columns\n')
					except AttributeError:
						pass
					# After check unmapped columns -> make a new dataframe based on the iamge ...
					if self.M is not None:
						ur = self._check_unmatched_rows(frame = frame)
						if ur == 0:
							return

					if self.M is not None:
						self.dtf1.loc[self.id1[self.M[:, 1]], "track_label"]   = self.dtf0.loc[self.id0[self.M[:, 0]], "track_label"].values
						self.dtf1.loc[self.id1[self.M[:, 1]], "lineage_label"] = self.dtf0.loc[self.id0[self.M[:, 0]], "lineage_label"].values
						self.dtf1.loc[self.id1[self.M[:, 1]], "track_length"]  = self.dtf0.loc[self.id0[self.M[:, 0]], "track_length"].values + dt
						

					self.dtf1.loc[self.id1[self.uC], "new_cell"]      = True
					self.dtf1.loc[self.id1[self.uC], "track_label"]   = self.max_label + np.arange(self.uC.shape[0], dtype = 'int')
					self.dtf1.loc[self.id1[self.uC], "lineage_label"] = self.max_label + np.arange(self.uC.shape[0], dtype = 'int')
					self.max_label                                    = self.max_label + self.uC.shape[0]
					
					

					for ii in self.uR:
						label  = self.dtf0.loc[self.id0[ii], "track_label"]
						length = self.dtf0.loc[self.id0[ii], "track_length"]
						if (length < 5) and (self.dtf0.loc[self.id0[ii], "lineage_label"] == label):
							self.uR[self.uR == ii] = -1
					
					self.uR = self.uR[self.uR >= 0]
					
					self.dtf0.loc[self.id0[self.uR], "mem"] = self.dtf0.loc[self.id0[self.uR], "mem"] + dt
					self.uR                       = self.uR[(self.dtf0.loc[self.id0[self.uR], "mem"] <= self.mem)]
					
					self.dtf0.loc[self.id0[self.uR], "frame"] = frame + dt
					self.dtf1     = pd.concat([self.dtf1, self.dtf0.loc[self.id0[self.uR]]], ignore_index = True)
					
					self.dtf1 = self.dtf1[self.dtf1["track_label"].isnull() == False]
					self.dtf1.loc[self.dtf1["mitosis"] == True, "mitosis"]     = False
					
					break


				self.dtf1.drop(columns = columns).to_csv(save_path, mode = "a", index = False, header = False)
				self.dtf0 = self.dtf1
				tf = time.time()
				log_file.write(f"Frame {frame:04}, {time.time()-time0} seconds\n")

	# Check unmatched rows ...
	def _check_unmatched_rows(self, frame):
		for ii in self.uR:
			if ii < 0:
				continue
			
			dtf_ii = self.dtf0.loc[self.id0[ii]]
			
			overlaps_ii = self.overlaps[ii, :]
			if not np.any(overlaps_ii > self.min_overlap):
				continue
			
			# Get the potential indices
			max_overlap       = np.max(overlaps_ii)
			merged_t1_idx     = np.where(overlaps_ii == max_overlap)[0][0]
			dtf1_merged       = self.dtf1.loc[self.id1[merged_t1_idx]]
			
			# Get the idx of any overlapping object ...
			overlaps_t1  = self.overlaps[:, merged_t1_idx]
			overlaps_idx = overlaps_t1 > self.min_overlap
			
			# All the merged objects
			merged_t0_all = np.where(overlaps_idx)[0]
			merged_t0_all = merged_t0_all[merged_t0_all != ii]
			
			# Get the idx of pre-merged object
			dtf0_merged       = self.dtf0.loc[self.id0[merged_t0_all]]
			
			# Alternating mem objects
			if dtf0_merged.shape[0] == 1:
				overlap = self._overlap(dtf_ii, dtf0_merged.iloc[0])
				if overlap > 0.7: # BENSON - this is arbitrary
					mem_ii = dtf_ii["mem"]
					mem_merged = dtf0_merged.iloc[0]["mem"]
					if dtf_ii["track_label"] == dtf_ii["lineage_label"]:
						if dtf0_merged.iloc[0]["track_label"] == dtf0_merged.iloc[0]["lineage_label"]:
							
							# This is the mapped object ...
							if mem_merged > 0:
								self.uR[self.uR == ii] = -1
								self.M[self.M[:, 0] == merged_t0_all, 0] = ii
								continue
							
							# This is the unmapped row ...
							if mem_ii > 0:
								self.uR[self.uR == ii] = -1
								continue
								
			# Conditions for preventing the alternating mergers ... 
			if dtf0_merged.shape[0] == 1:
				if max_overlap > 0.4: 
					length    = dtf0_merged["track_length"].values[0]
					length_uR = dtf_ii["track_length"]
					if dtf0_merged["track_label"].values[0] == dtf0_merged["lineage_label"].values[0]:
						if length_uR > length:
							self.uR[self.uR == ii] = -1
							self.M[self.M[:, 0] == merged_t0_all, 0] = ii
							continue
					
				# Check how long the track is for dtf0_merged
				
			# Only check these if the max_overlap is not that significant?
			pii = dtf_ii["perimeter"]
			p0  = dtf0_merged["perimeter"].values
			p1  = dtf1_merged["perimeter"]
			if p1 / (pii + np.sum(p0)) < 0.7:
				continue

			aii = dtf_ii["area"]
			a0  = dtf0_merged["area"].values
			a1  = dtf1_merged["area"]
			if a1 / (aii + np.sum(a0)) < 0.7:
				continue
			
			bbox_merged = dtf1_merged.bbox
			# --------------------------------------------------------------------
			# --------------------------------------------------------------------
			# Create  a new image with the "label" being the idx of M[:, 0], uR
			# --------------------------------------------------------------------
			# --------------------------------------------------------------------
			tmp = np.zeros((bbox_merged[2] - bbox_merged[0], 
							bbox_merged[3] - bbox_merged[1]), dtype = np.uint16);
			
			x = self._centroid_cropped(dtf_ii, bbox_merged[0], tmp.shape, dim = 0)
			y = self._centroid_cropped(dtf_ii, bbox_merged[1], tmp.shape, dim = 1)
			tmp[x, y] = ii
			for cntr, (index, row) in enumerate(dtf0_merged.iterrows()):
				x = self._centroid_cropped(row, bbox_merged[0], tmp.shape, dim = 0)
				y = self._centroid_cropped(row, bbox_merged[1], tmp.shape, dim = 1)
				tmp[x, y] = merged_t0_all[cntr]
			
			tmp_merged = np.zeros(tmp.shape, dtype = np.uint8)
			tmp_merged[self._coords(dtf1_merged, bbox_merged[0], tmp.shape, dim = 0),
					   self._coords(dtf1_merged, bbox_merged[1], tmp.shape, dim = 1)] = 1
			
			xy     = np.where(tmp > 0)
			interp = interpolate.NearestNDInterpolator(np.transpose(xy), tmp[xy])
			B      = interp(*np.indices(tmp.shape))
			B[tmp_merged == 0] = 0
				
			reg    = regionprops(B)
			reg    = [r for r in reg if r.area > self.min_area]

			# Create the new dataframe
			if len(reg) <= 1:
				self.uR[self.uR == ii] = -1
				continue
			

			dtf_unmerged = pd.DataFrame([Cell.from_regionprops(r, bbox = bbox_merged, frame = frame + self.dt) for r in reg])
			
			# Set the mean gfp properly
			dtf_unmerged["mean_gfp"] = dtf1_merged["mean_gfp"]

			# --------------------------------------------------------------------
			# dtf1_merged.name is the index of the first object
			index     = dtf1_merged.name
			max_index = np.max(self.dtf1.index)
			dtf_unmerged.rename(index = {ii: max_index + ii for ii in dtf_unmerged.index[1:]}, inplace = True)
			dtf_unmerged.rename(index = {0:index}, inplace = True)
			dtf_unmerged["merged"]    = True
			
			try:
				self.dtf1.loc[index] = dtf_unmerged.loc[index]
			except KeyError as err:
				display(len(reg))
				display(index)
				display(dtf_unmerged)
				raise KeyError
			
			self.dtf1 = pd.concat([self.dtf1, dtf_unmerged.iloc[1:]])
			self.id1  = np.append(self.id1, dtf_unmerged.index[1:])
			
			new_pairs = np.zeros((dtf_unmerged.shape[0], 2), dtype = 'int')
			for cntr, (index, row) in enumerate(dtf_unmerged.iterrows()):
				new_pairs[cntr, 0] = row["label"]
				new_pairs[cntr, 1] = np.where(self.id1 == index)[0][0]
			
			self.M[np.in1d(self.M[:, 0], merged_t0_all), :] = -1
			self.M = np.append(self.M, new_pairs, axis = 0)
			self.uR[np.in1d(self.uR, merged_t0_all)] = -1
			self.uR[self.uR == ii] = -1
		
		self.M   = self.M[self.M[:, 0] >= 0, :]
		self.uR  = self.uR[self.uR >= 0]
		return None
