=============
Release Notes
=============


Viziphant 0.4.0 release notes
*****************************
New functionality and features
------------------------------
* Added new functionality to visualize complex sets of (spike) patterns as hypergraphs (#75)

Bug fixes
---------
* Resolved a bug in the `plot_ue` function that addressed issues with joining axes in the plot (#72)

Miscellaneous
-------------
* Updated the copyright statement to reflect the latest changes (#68)
* Enhance security of actions (#69)
* Install mirror to gitlab (#67)
* In Readthedocs, deprecated the configuration key `build.image` in favor of `build.os`  (#71)

Selected dependency changes
---------------------------
bokeh>=3.0.0
holoviews>=1.16.0
networkx>=3.0.0


Viziphant 0.3.0 release notes
*****************************

Documentation
-------------
* Fixed math display for equations in docs by using mathjax as renderer #59
* Remove title "Viziphant" from navigation bar #60
* Fix documentation, remove version cap on sphinxcontrib-bibtex, add new citation style   #57
* Fix navigation bar not showing up in documentation #64

New functionality and features
------------------------------
* Add parameter to exclude labels from plotting in add_event function. #50
* Add limit check to add_event function to prevent labels outside bounds of figure. #49
* GPFA function plot_dimensions_vs_time passing neo.Events to the plotting function to make it more robust and flexible #52

Bug fixes
---------
* Fix bug in add_events where only the last event was plotted. #53
* Fix bug with newer versions of matplotlib by replacing gca() with add_subplot(). #56

Miscellaneous
-------------
* CI: update GitHub actions commands, remove deprecated commands  #58
* Add first unit test for ASSET plot_synchronous_events #63

Selected dependency changes
---------------------------
* Update python version, drop support for python 3.6 and 3.7 #51 #62
* Update to quantities >0.14.0 #61


Viziphant 0.2.0 release notes
*****************************

Documentation
-------------
* Documentation revised in style, logo was added

New functionality and features
------------------------------
* New `patterns` module to display spike patterns by elephant modules, such as SPADE or CAD (https://github.com/INM-6/viziphant/pull/35)
* New function to display the spike contrast measure (https://github.com/INM-6/viziphant/pull/34)
* Added function to display rate trajectories plotted by the GPFA module (https://github.com/INM-6/viziphant/pull/37)

Bug fixes
---------
* Bug fixes for the ISI and Unitary Event plots (https://github.com/INM-6/viziphant/pull/31) and (https://github.com/INM-6/viziphant/pull/32).

Miscellaneous
-------------
* Continuous Integration (CI) was moved to github actions (https://github.com/INM-6/viziphant/pull/41)

Selected dependency changes
---------------------------
* `elephant>=0.9.0`


Viziphant 0.1.0 release notes
*****************************

This release constitutes the initial Viziphant release.
