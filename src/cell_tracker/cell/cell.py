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
from skimage.measure     import regionprops
from dataclasses         import dataclass, field
from typing              import List

import numpy as np
import pandas as pd

# This dataclass is just to keep everything organized and create 
# lists of objects from labeled images.
@dataclass
class Cell:
    label: int             = 0
    frame: int             = 0
    x:     float           = None
    y:     float           = None
    area:  int             = None
    
    colony_label:   int    = None
    mean_gfp:       float  = None
    perimeter:      float  = None
    bbox:        List[int] = field(default_factory = list)
    xcoords:     List[int] = field(default_factory = list)
    ycoords:     List[int] = field(default_factory = list)
    
    # Extra fields for cell tracking
    lineage_label: int   = None
    track_label:   int   = None
    mitosis:       int   = False
    mother_label:  int   = None
    merged:        bool  = None
    new_cell:      int   = False
    mem:           int   = 0
    track_length:  int   = 0

    # Returns the Cell object from skimage regionprops
    @classmethod
    def from_regionprops(cls, reg = None, img_lbl = None, frame = None, bbox = None, img_fluo = None):
        if reg is None and img_lbl is None:
            raise ValueError("Need to provide a labeled image or regionprops object")
        
        if img_lbl is not None:
            reg = regionprops(img_lbl)
    
        label     = reg.label
        x, y      = reg.centroid
        area      = reg.area
        coords    = reg.coords

        if bbox is not None:
            x, y = x + bbox[0], y + bbox[0]

            coords[:, 0] = coords[:, 0] + bbox[0]
            coords[:, 1] = coords[:, 1] + bbox[1]

        bbox      = reg.bbox
        perimeter = reg.perimeter


        # computing the mean gfp - could just save the pixel values instead
        mean_gfp = None
        if img_fluo is not None:
            mean_gfp = np.mean(img_fluo[coords[:, 0], coords[:, 1]])

        return cls(label, frame, x, y, area,    perimeter   = perimeter, 
                                                bbox        = bbox, 
                                                xcoords     = coords[:, 0].tolist(), 
                                                ycoords     = coords[:, 1].tolist(),
                                                mean_gfp    = mean_gfp)
    @classmethod
    def from_dataframe(cls, row_dict):
        return pd.DataFrame([cls(**row_dict)])

    def to_pandas(self, **kwargs):
        return pd.DataFrame([self])
    
