
from cell_tracker.experiment_info 	import get_experiment_path
from cell_tracker.segmentation 		import FluoSegmenter
from cell_tracker.tracker 			import CellTrackerLauncher
from cell_tracker.mitosis 			import MitosisTrackerFluo
from os.path 						import join
import os

class Main(FluoSegmenter):

	_defaults = {
		'mitosis_filename': MitosisTrackerFluo._defaults.get('mitosis_filename'),
		'channel_name': 'fluorescence',
		'image_folder': 'fluorescence-segmented',
		'redo_mitosis':      True,
		'redo_segmentation': False,
		'redo_tracking':     True,
		'tf': 600,
		't0': 0,
		'dt': 1,
		'tracker_parms': CellTrackerLauncher._defaults.get('tracker_parms'),
	}

	def __init__(self, root, image_folder, **kwargs):
		super().__init__(**kwargs)

		for k, v in Main._defaults.items():
			if kwargs.get(k) is not None:
				setattr(self, k, kwargs.get(k))
				continue
			setattr(self, k, v)

		self.root = root
		self.load_dir = join(root, image_folder)

	def run(self, ):

		frames = range(self.t0, self.tf, self.dt)
		
		# Segment Fluorescence channel
		self.segment_fluorescence(load_dir = self.load_dir, frames = frames, redo = self.redo_segmentation)

		# Get the mitosis xyt
		if not os.path.isfile(join(self.root, self.mitosis_filename)) or self.redo_mitosis:
			MitosisTrackerFluo(	root 			 = self.root, 
								load_dir 		 = f'{self.load_dir}-mitosis',
								mitosis_filename = self.mitosis_filename).run()

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
	root = get_experiment_path(date = '230202', well = 3)

	Main(root = root, image_folder = 'fluorescence').run()
