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

import multiprocessing as mp
from multiprocessing import Pool
import sys

# This class just helps split up tasks among cpus to speed up codes
class ParallelProcessor:

	def __init__(self, **kwargs):
		for k, v in kwargs.items():
			setattr(self, k, v)

	def run_series(self, function, iterations, cpus = None, verbose = False):
		max_cpus = mp.cpu_count()

		if cpus is None:
			cpus = max_cpus
			
		if cpus > max_cpus:
			cpus = max_cpus

		if verbose:
			print(f"Running {cpus} processes")
		
		with Pool(cpus) as p: 
			results = p.map(function, iterations)

		if verbose:
			print(f"Finished parallel processes")
		
		return results

	def set_defaults(self, defaults):
		for k, v in defaults.items():
			if not hasattr(self, k):
				setattr(self, k, v)


		
