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
import sys, os, json

from datetime 		 import date
from skimage.measure import regionprops
from os.path 		 import join
from . 				 import CellTracker, ParallelProcessor, Cell, skio, np, pd

class CellTrackerLauncher(ParallelProcessor):

	_defaults = {
		'root': 			'Z:/Analysis/InScoper/221029_H2B_Live/221029_H2B_Live_Media/well02/stitched_images',
		'image_folder': 	join('phase_inferenced_processed', '220908_200k_m2_w2_thresh2_python'),
		'mitosis_filename': 'xyt_divisions_from_3dunet_python_new.csv',
		'channel_name': 	'inferenced',
		'image_filename':   'image-{:04d}.tif',
		'tracker_parms': {
			    'mem':         		5,
			    'max_disp':    		20,
			    'min_overlap': 		0.05,
			    'limits':      		[],
			    'min_track_length': 25,
			    'min_object_size': 	80,
		},
		'tmp_dir': 		   	'tmp',
		'min_object_size': 	80,
		'dt': 				1,
		't0': 				0,
		'tf': 				700,
	}

	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		for key, val in CellTrackerLauncher._defaults.items():
			if not hasattr(self, key):
				setattr(self, key, val)

	def run(self, save_filename = None, redo = False, function = None):

		today = date.today().strftime("%y%m%d")

		if function is None:
			function = self.update
		
		# default filename
		if save_filename is None:
			save_filename = f'tracks-{today}-{self.channel_name}.csv'

		# save the tracking parameters
		parms_filename = f'{save_filename[0:-4]}-parameters.json'
		with open(join(self.root, parms_filename), 'w') as file_id:
			parms 						= self.tracker_parms.copy()
			parms['root']             	= self.root
			parms['image_folder']     	= self.image_folder
			parms['mitosis_filename'] 	= self.mitosis_filename
			parms['channel_name']     	= self.channel_name
			parms['min_object_size']  	= self.min_object_size
			parms['t0'] 				= self.t0
			parms['tf'] 				= self.tf
			parms['dt'] 				= self.dt
			json.dump(parms, file_id)


		# Check if tracks already exist ...
		if os.path.isfile(join(self.root, save_filename)) and not redo:
			print("Tracking Results already exist")
			return

		self.shape = skio.imread(join(self.root, self.image_folder, self.image_filename.format(0))).shape
		
		print("making the dfs")
		os.makedirs(join(self.root, self.tmp_dir), exist_ok = True)
		self.run_series(function, range(self.t0, self.tf), cpus = 20)

		# run the cell tracker ...
		self.tracker    = CellTracker(root 				= self.root, 
									 load_dir 			= join(self.root, self.tmp_dir), 
									 img_shape 			= self.shape, 
									 mitosis_filename 	= self.mitosis_filename, **self.tracker_parms)

		print("launching the tracker")
		self.tracker._run(t0 = self.t0, tf = self.tf - 1, dt = self.dt, save_path = join(self.root, save_filename))

	def get_DataFrame(self, frame, **kwargs):
		load_path = join(self.root, self.image_folder, self.image_filename.format(frame))
		img = skio.imread(load_path)
		reg = regionprops(img)
		df  = pd.DataFrame([Cell.from_regionprops(reg = r, frame = frame, img_fluo = None) for r in reg if r.area > self.min_object_size])
		return df

	def update(self, frame, **kwargs):
		df = self.get_DataFrame(frame, **kwargs)
		df.to_pickle(join(self.root, self.tmp_dir, f'cells-{frame:04}.pkl'))

