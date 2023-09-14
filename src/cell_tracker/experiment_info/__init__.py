def get_experiment_path(date, well = 0, root = 'Z:/Analysis/InScoper', **kwargs):
	
	if date in ['221029', '221104', '221110']:
		root = f'{root}/{date}_H2B_Live/{date}_H2B_Live_Media'
		
	if date in ['230202']:
		root = f'{root}/{date}_H2B_Fluo'
		
	if date in ['230427', '230511', '230518']:
		root = f'{root}/{date}_H2B_Live'
		
	if date in ['230525']:
		root = f'{root}/{date}_H2B-partial_Live/{date}_H2B-partial_Live'

	root = f'{root}/well{well:02}/stitched_images'
	return root


