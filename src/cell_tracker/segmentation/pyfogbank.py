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
from skimage.measure    import regionprops
from scipy              import ndimage
from .                  import cv2, np

class FogBank:
    def __init__(self, img, 
                 min_size        = 15, 
                 min_object_size = 50, 
                 erode_size      = 3, 
                 num_levels      = 50, **kwargs):
        self.img             = img
        self.min_size        = min_size
        self.min_object_size = min_object_size
        self.erode_size      = (erode_size, erode_size)
        self.num_levels      = num_levels

    # Apply the percentile threshold to get the fg
    def _get_foreground(self, level):
        return cv2.threshold(self.dist, level, 255, cv2.THRESH_BINARY)[1].astype(np.uint8)
        
    def _remove_small_objects(self, fg, **kwargs):
        [num_labels, cc, stats]  = cv2.connectedComponentsWithStats(fg)[0:3]
        
        if num_labels == 1:
            return fg
        
        # Get labels to remove
        labels_to_remove = [label for label in range(1, num_labels) 
                                  if stats[label, cv2.CC_STAT_AREA] < self.min_size]

        cc[np.in1d(cc.flatten(), labels_to_remove).reshape(cc.shape)] = 0
        return cv2.connectedComponents((cc > 0).astype(np.uint8))[1]
        
    # Add in percentile levels here
    def _set_percentile_levels(self, **kwargs):
        levels = np.linspace(1, 0, self.num_levels)
        
        for cntr, ii in enumerate(np.linspace(1, 0, self.num_levels)):
            levels[cntr] = np.percentile(self.dist[self.dist > 0], 100 * ii)
        
        self.levels = np.flip(np.unique(levels))


    def run(self, img_to_transform = None, **kwargs):
        
        # erode the mask here
        kern         = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.erode_size)
        self.img_bin = cv2.erode(self.img, kern, iterations = 2)
        
        # Distance transform
        if img_to_transform is None:
            img_to_transform = self.img_bin
        self.dist    = cv2.distanceTransform(img_to_transform, cv2.DIST_L2, 5)
        
        # This is the "fog levels"
        self._set_percentile_levels()
        while True:
            try:
                self.levels      = self.levels[1:]
                fg0              = self._get_foreground(self.levels[0])
                cc0              = self._remove_small_objects(fg0)
                self.working_img = cc0
                max_id           = np.max(cc0[cc0 > 0])
            except ValueError:
                continue
            break

        # Go through the fog levels
        for level in self.levels[1:]:
            
            # Dropped one level
            fg1      = self._get_foreground(level)
            max_id   = self._fog(fg0, cc0, fg1, max_id)
            fg0      = fg1
        
        self._fill()
        # Get the regionprops
        working_img  = self.working_img.copy()
        reg          = regionprops(self.working_img)
        
        # Labels to remove
        labels_to_remove = [r.label for r in reg if r.area < self.min_object_size]
        
        self.working_img[np.in1d(self.working_img, labels_to_remove).reshape(self.working_img.shape)] = 0
        self._fill()

        if np.max(self.working_img) < 2**16:
            self.working_img = self.working_img.astype(np.uint16)

        return self.working_img
        
    def _fog(self, fg0, cc0, fg1, max_id, **kwargs):
        # Lower the fog level and update the working image
        # This is where the slowdown occurs !!! **** BENSON ****
        B                         = ndimage.maximum_filter(self.working_img, 3)
        B[self.working_img != 0]  = self.working_img[self.working_img != 0]
        self.working_img          = B * (fg1 > 0)

        # Go to the new objects
        # Taking the new working_img and removing it from fg1
        fg_orig                   = fg1.copy()
        fg1[self.working_img > 0] = 0
        cc1                       = self._remove_small_objects(fg1)
        
        new_ids       = np.max(cc1)
        cc1[cc1 != 0] = cc1[cc1 != 0] + max_id + 1
        max_id        = max_id + new_ids
        
        # Assign new peaks to the seed image
        self.working_img[self.working_img == 0] = cc1[self.working_img == 0]
        
        # Assign the unassigned pixels to their corresponding values
        # This is where the slowdown occurs !!! **** BENSON ****
        B                         = ndimage.maximum_filter(self.working_img, 3)
        B[self.working_img != 0]  = self.working_img[self.working_img != 0]
        self.working_img          = B * (fg_orig > 0)
        return max_id

    def _fill(self, **kwargs):
        img_sum = np.sum((self.working_img > 0) * self.img)
        for ii in range(5):
            B                        = ndimage.maximum_filter(self.working_img, 3)
            B[self.working_img != 0] = self.working_img[self.working_img != 0]
            self.working_img         = B * (self.img > 0)
            
            img_sum_tmp = np.sum((self.working_img > 0) * self.img)
            if  img_sum_tmp == img_sum:
                break
            img_sum = img_sum_tmp