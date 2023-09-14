# iPSC_tracking

# create the environment

### This is for Linux
```bash
# Make sure you have the right python version downloaded
python --version

# Create the virtual environment
python -m venv venv

# activate virtual environment
source venv/bin/activate

# update package manager and download required packages ... 
python -m pip install --upgrade pip
pip install -r requirements.txt
```
### This is for Windows
```
python -m venv venv
venv/Scripts/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

# Segmenting a fluorescence image

To segment the fluorescence channel, you need to use the routines Threshold and FogBank in the segmentation folder. 

If you already have a thresholded mask, then only run the fogbank section below with the mask as input. 

FYI - You can also use a different image to do the distance transform by specifying img_to_transform in the Fogbank.run routine. This can be the fluorescence image. The image should be a uint8.

```python
from cell_tracker.segmentation import Fogbank, Threshold
import skimage.io as skio

# Load the image at specified path
img        = skio.imread(path_to_image)

# Threshold the image
thresh_parms = {
    'sigma':        1,
    'boxcar_size':  26,
    'thresh':       1e2,

}
img_thresh = Threshold(img, **thresh_parms).run()

# Segment the image 
fb_parms = {
    'min_size':         10, 
    'min_object_size':  80, 
    'erode_size':       3, 

}
img_seg    = FogBank(img, **fb_parms).run(img_to_transform = None)

```

# Launching Cell Tracker
This section shows how to launch the cell tracker on a collection of images ...
```python
tracker_parms = {
    'mem':         		5,    # maximum amount of time an object can be unmapped
    'max_disp':    		20,   # max separation
    'min_overlap': 		0.05, # min mask overlap
    'limits':      		[],   # this crops the image
    'min_object_size': 	80,   # minimum object size
}

kwargs = {
    'root': '/mnt/x/exp0.0', # root is the relative path for all files. 
    'image_folder': 'phase_inferenced_processed', # Folder with the labeled images.
    'mitosis_filename': 'xyt_divisions.pkl', # filename with mitosis xyt.
    'tracker_parms': tracker_parms, # Tracking parameters
    't0': 0, # Initial frame
    'dt': 1, # time resolution - be careful with mitosis detection time resolution
    'tf': 700, # final frame
}

# This will run the cell tracker
from cell_tracker.tracker import CellTrackerLauncher as CTL
ctl = CTL(**kwargs)
ctl.run(save_filename = 'tracks')
```




# NIST Open-Source Software Repository Template

Use of GitHub by NIST employees for government work is subject to
the [Rules of Behavior for GitHub][gh-rob]. This is the
recommended template for NIST employees, since it contains
required files with approved text. For details, please consult
the Office of Data & Informatics' [Quickstart Guide to GitHub at
NIST][gh-odi].

Please click on the green **Use this template** button above to
create a new repository under the [usnistgov][gh-nst]
organization for your own open-source work. Please do not "fork"
the repository directly, and do not create the templated
repository under your individual account.

The key files contained in this repository -- which will also
appear in templated copies -- are listed below, with some things
to know about each.

---

## README

Each repository will contain a plain-text [README file][wk-rdm],
preferably formatted using [GitHub-flavored Markdown][gh-mdn] and
named `README.md` (this file) or `README`.

Per the [GitHub ROB][gh-rob] and [NIST Suborder 1801.02][nist-s-1801-02],
your README should contain:

1. Software or Data description
   - Statements of purpose and maturity
   - Description of the repository contents
   - Technical installation instructions, including operating
     system or software dependencies
1. Contact information
   - PI name, NIST OU, Division, and Group names
   - Contact email address at NIST
   - Details of mailing lists, chatrooms, and discussion forums,
     where applicable
1. Related Material
   - URL for associated project on the NIST website or other Department
     of Commerce page, if available
   - References to user guides if stored outside of GitHub
1. Directions on appropriate citation with example text
1. References to any included non-public domain software modules,
   and additional license language if needed, *e.g.* [BSD][li-bsd],
   [GPL][li-gpl], or [MIT][li-mit]

The more detailed your README, the more likely our colleagues
around the world are to find it through a Web search. For general
advice on writing a helpful README, please review
[*Making Readmes Readable*][18f-guide] from 18F and Cornell's
[*Guide to Writing README-style Metadata*][cornell-meta].

## LICENSE

Each repository will contain a plain-text file named `LICENSE.md`
or `LICENSE` that is phrased in compliance with the Public Access
to NIST Research [*Copyright, Fair Use, and Licensing Statement
for SRD, Data, and Software*][nist-open], which provides
up-to-date official language for each category in a blue box.

- The version of [LICENSE.md](LICENSE.md) included in this
  repository is approved for use.
- Updated language on the [Licensing Statement][nist-open] page
  supersedes the copy in this repository. You may transcribe the
  language from the appropriate "blue box" on that page into your
  README.

If your repository includes any software or data that is licensed
by a third party, create a separate file for third-party licenses
(`THIRD_PARTY_LICENSES.md` is recommended) and include copyright
and licensing statements in compliance with the conditions of
those licenses.

## CODEOWNERS

This template repository includes a file named
[CODEOWNERS](CODEOWNERS), which visitors can view to discover
which GitHub users are "in charge" of the repository. More
crucially, GitHub uses it to assign reviewers on pull requests.
GitHub documents the file (and how to write one) [here][gh-cdo].

***Please update that file*** to point to your own account or
team, so that the [Open-Source Team][gh-ost] doesn't get spammed
with spurious review requests. *Thanks!*

## CODEMETA

Project metadata is captured in `CODEMETA.yaml`, used by the NIST
Software Portal to sort your work under the appropriate thematic
homepage. ***Please update this file*** with the appropriate
"theme" and "category" for your code/data/software. The Tier 1
themes are:

- [Advanced communications](https://www.nist.gov/advanced-communications)
- [Bioscience](https://www.nist.gov/bioscience)
- [Buildings and Construction](https://www.nist.gov/buildings-construction)
- [Chemistry](https://www.nist.gov/chemistry)
- [Electronics](https://www.nist.gov/electronics)
- [Energy](https://www.nist.gov/energy)
- [Environment](https://www.nist.gov/environment)
- [Fire](https://www.nist.gov/fire)
- [Forensic Science](https://www.nist.gov/forensic-science)
- [Health](https://www.nist.gov/health)
- [Information Technology](https://www.nist.gov/information-technology)
- [Infrastructure](https://www.nist.gov/infrastructure)
- [Manufacturing](https://www.nist.gov/manufacturing)
- [Materials](https://www.nist.gov/materials)
- [Mathematics and Statistics](https://www.nist.gov/mathematics-statistics)
- [Metrology](https://www.nist.gov/metrology)
- [Nanotechnology](https://www.nist.gov/nanotechnology)
- [Neutron research](https://www.nist.gov/neutron-research)
- [Performance excellence](https://www.nist.gov/performance-excellence)
- [Physics](https://www.nist.gov/physics)
- [Public safety](https://www.nist.gov/public-safety)
- [Resilience](https://www.nist.gov/resilience)
- [Standards](https://www.nist.gov/standards)
- [Transportation](https://www.nist.gov/transportation)

---

[usnistgov/opensource-repo][gh-osr] is developed and maintained
by the [opensource-team][gh-ost], principally:

- Gretchen Greene, @GRG2
- Yannick Congo, @faical-yannick-congo
- Trevor Keller, @tkphd

Please reach out with questions and comments.

<!-- References -->

[18f-guide]: https://github.com/18F/open-source-guide/blob/18f-pages/pages/making-readmes-readable.md
[cornell-meta]: https://data.research.cornell.edu/content/readme
[gh-cdo]: https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-code-owners
[gh-mdn]: https://github.github.com/gfm/
[gh-nst]: https://github.com/usnistgov
[gh-odi]: https://odiwiki.nist.gov/ODI/GitHub.html
[gh-osr]: https://github.com/usnistgov/opensource-repo/
[gh-ost]: https://github.com/orgs/usnistgov/teams/opensource-team
[gh-rob]: https://odiwiki.nist.gov/pub/ODI/GitHub/GHROB.pdf
[gh-tpl]: https://github.com/usnistgov/carpentries-development/discussions/3
[li-bsd]: https://opensource.org/licenses/bsd-license
[li-gpl]: https://opensource.org/licenses/gpl-license
[li-mit]: https://opensource.org/licenses/mit-license
[nist-code]: https://code.nist.gov
[nist-disclaimer]: https://www.nist.gov/open/license
[nist-s-1801-02]: https://inet.nist.gov/adlp/directives/review-data-intended-publication
[nist-open]: https://www.nist.gov/open/license#software
[wk-rdm]: https://en.wikipedia.org/wiki/README
