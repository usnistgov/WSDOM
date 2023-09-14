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
import numpy as np

class CellTrackerFactory:

    _defaults = {
            'mem':                 0,
            'max_disp':            20,
            'min_overlap':         0.2,
            'mitosis_time_cutoff': 6,
            'min_area':            40,
    }

    def set_defaults(self, **kwargs):
        for k, v in self._defaults.items():
            if kwargs.get(k) is not None:
                setattr(self, k, kwargs.get(k))
                continue
            setattr(self, k, v)
        
    def __init__(self, **kwargs):
        self.set_defaults(**kwargs)

    @staticmethod
    def _centroid(dtf, **kwargs):
        return dtf[["x", "y"]].values
    
    # Returns linear pixellist from x, y coordinates
    def _pixellist(self, x, y = None):
        if y is None:
            return x[:, 0] * self.shape[0] + x[:, 1]
        return np.array(x) * self.shape[0] + np.array(y)

    def _remove_single_matches(self, idx, id0, id1):
        # rows that map to a single column
        deg0 = np.sum(idx, axis = 1) == 1
        # columns that map to a single row
        deg1 = np.sum(idx, axis = 0) == 1

        id0_to_remove = id0[deg0]
        id1_to_remove = np.zeros(id0_to_remove.shape[0], dtype = 'int')
        for ii in range(id0_to_remove.shape[0]):
            id1_to_remove[ii] = id1[idx[id0_to_remove[ii], :]]

        single_matches = np.in1d(id1_to_remove, id1[deg1])

        # These are my initial pairs
        id0_to_remove = id0_to_remove[single_matches]
        id1_to_remove = id1_to_remove[single_matches]

        return id0_to_remove, id1_to_remove
    
    @staticmethod
    def create_labeled_image(dtf, shape, column = "track_label", min_size = 50):
        img = np.zeros(shape, dtype = np.uint16)
        for index, row in dtf.iterrows():
            if not row.isna()["track_label"]:
                img[row["xcoords"], [row["ycoords"]]] = row[column]
        return img
    
    # returns the coords of a cropped image
    def _coords(self, row, x0, shape, dim = 0):
        if dim == 0:
            coords = np.array(row["xcoords"]) - x0
            coords[coords < 0] = 0
            coords[coords > shape[0]-1] = shape[0] - 1
        if dim == 1:
            coords = np.array(row["ycoords"]) - x0
            coords[coords < 0] = 0
            coords[coords > shape[1]-1] = shape[1] - 1
        return coords
    
    # This gets the centroid of a cropped image
    def _centroid_cropped(self, row, x0, shape, dim = 0):
        if dim == 0:
            x = int(row["x"] - x0)
            if x < 0:
                x = 0
            if x > shape[0] - 1:
                x = shape[0] - 1
        if dim == 1:
            x = int(row["y"] - x0)
            if x < 0:
                x = 0
            if x > shape[1] - 1:
                x = shape[1] - 1
        return x
    
    # This overlap is IoU
    def _overlap(self, dtf0, dtf1):
        plist0 = self._pixellist(dtf0["xcoords"], dtf0["ycoords"])
        plist1 = self._pixellist(dtf1["xcoords"], dtf1["ycoords"])
        return np.sum(np.in1d(plist0, plist1)) / len(np.unique(np.append(plist0, plist1)))
    
    def _overlap_ij_pixellist(self, plist0, plist1):
        return np.sum(np.in1d(plist0, plist1)) / plist0.shape[0]
    
    # This overlap is going to be the in1d(0, 1) / 0.shape[0]
    def _overlap_ij(self, dtf0, dtf1):
        plist0 = self._pixellist(dtf0["xcoords"], dtf0["ycoords"])
        plist1 = self._pixellist(dtf1["xcoords"], dtf1["ycoords"])
        return self._overlap_ij_pixellist(plist0, plist1)
    
    def _bbox(self, x, y = None):
        return (np.min(x), np.max(x), np.min(y), np.max(y))

# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
