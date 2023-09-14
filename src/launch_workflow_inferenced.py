from cell_tracker.experiment_info import get_experiment_path
from cell_tracker.segmentation import InferencedSegmenter
from cell_tracker.tracker import CellTrackerLauncher
from cell_tracker.mitosis import MitosisDetector
from os.path import join
import os, sys

class Main(InferencedSegmenter):

	_defaults = {
		'threshold': 2,
		'mitosis_filename': getattr(MitosisDetector, '_default_save_filename'),
		'image_folder': join('phase-inferenced-processed', '220908-200k-eroded-w2-m3-thresh2'),
		'redo_mitosis':      True,
		'redo_segmentation': False,
		'redo_tracking':     True,
		'tf': 700,
		't0': 0,
		'dt': 1,
		'tracker_parms': CellTrackerLauncher._defaults.get('tracker_parms'),
	}

	def __init__(self, root, **kwargs):
		super().__init__(**kwargs)

		for k, v in Main._defaults.items():
			if kwargs.get(k) is not None:
				setattr(self, k, kwargs.get(k))
				continue
			setattr(self, k, v)

		self.channel_name = f'inferenced-{self.image_folder}'
		self.root         = root

	def run(self, ):

		frames = range(self.t0, self.tf, self.dt)
		
		# postprocess inferenced results
		self.segment_inferenced(root = self.root, frames = frames, redo = self.redo_segmentation)

		# get the mitosis xyt from 3D U-Net
		if not os.path.isfile(join(self.root, self.mitosis_filename)) or self.redo_mitosis:
			MitosisDetector(root 			 = self.root, 
							mitosis_filename = self.mitosis_filename).run(t0 = self.t0, tf = self.tf)

		# launch the cell_tracker
		kwargs = {k: v for k, v in CellTrackerLauncher._defaults.items()}
		for k, v in CellTrackerLauncher._defaults.items():
			if hasattr(self, k):
				kwargs[k] = getattr(self, k)
				if kwargs.get(k) is not None:
					continue
			kwargs[k] = v
		CellTrackerLauncher(**kwargs).run(redo = self.redo_tracking)

if __name__ == "__main__":
	root = get_experiment_path(date = '221029', well = 2)

	Main(root = root).run()
