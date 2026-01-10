# ALMA Visibility Modelling - Peter Knowlton
This repository in forked from Emily Carver's ALMA Visibility Modelling repository, with her original README below. The "main" branch is kept up to date with her original repository, while the "pfk-work" branch contains my contributions.

# ALMA Visibility Modelling - Emily Carver
This repository contains code and other documentation I wrote during the course of my co-op at DAO over the summer of 2025. My project allowed me to learn methods of image synthesis, and compare that to analysis performed directly with the data by testing visibility modelling on galaxies observed by ALMA. To end the term, I presented my work to my colleagues at HAA during Science Tea. A copy of my presentation slides is included in this repo.

When visibilities are imaged for visualization and analysis, the process of imaging introduces artifacts such as the resolution limit imposed by the synthesized beam. Rather than bringing visibilities into the image plane for analysis, it is possible to bring a model into the visibility space and perform analysis there. Visibility modelling techniques can allow for sub-beam structure to be recovered, without the sensitivity trade-off that comes with increased resolution when imaging. Visibility modelling has been utilized in the study of protoplanetary disks, where it has been used to investigate disk features such as gaps, rings, and inner cavities in great detail. However, this same technique has not been applied to structures in other systems, such as galaxies. 

This code was written to test visibility modelling on the starburst ring at the center of the nearby galaxy NGC 3351. See [Sun et al. (2024)](https://ui.adsabs.harvard.edu/abs/2024ApJ...967..133S/abstract) for the data used in this example and for an example of how the measurements provided by visibility modelling can allow us to study the nature of star formation. 

This repo contains two folders, which contain code used in two different implementations of visibility modelling. 

The radial_version folder contains code used to fit axisymmetric models defined by a radial brightness distribution. This method was tested with observations of the inner star-forming ring in NGC 3351, and produced fit models as well as products useful in the visualization of those fits. A README file in that folder goes into greater detail on that tool. 

The 2d_model_version folder contains code designed to fit models defined in a two-dimensional plane, thus allowing a greater freedom in the complexity of the model. During my co-op term I was not able to get this version of the code running as expected, but it is provided to show generally what this use case would look like. A README file in that folder goes into greater detail on what is included. 

Here is a list of helpful readings that I referenced often during my project:

*Galario*
- The [initial galario paper](https://ui.adsabs.harvard.edu/abs/2018MNRAS.476.4527T/abstract) is itself useful for understanding the expected use case of the tool and why certain limitations are in place (ex. small-field imaging, single channel with narrow bandwidth)
- Galario has been further developed since this publication, so the [docs](https://mtazzari.github.io/galario/) are important to reference as well. These docs have a simple example of combining galario with emcee to perform fittings that I built my tool off of. 

*Existing visibility modelling efforts*
- For an example of visibility modelling successfully applied to the study of protoplanetary disks, read [Long et al. (2018)](https://ui.adsabs.harvard.edu/abs/2018ApJ...869...17L/abstract). 
- While it is designed for specifically axisymmetric disk structures, the [Frankenstein](https://ui.adsabs.harvard.edu/abs/2020MNRAS.495.3209J/abstract) paper is very informative in terms of understanding how visibility fitting works. 

*Working with visibilities outside CASA*
- The u and v coordinates of the measured visibilities need to be taken out of the CASA measurement sets in order to be used with galario. The [uvplot package](https://uvplot.readthedocs.io/en/latest/) provides a straightforward way to do this (it just uses the CASAtools table methods, but in a very user-friendly way). 
- In order to image the residual visibilties, the calculated residual visibilities must be moved into a Measurement Set. The best method I found was to simply make a copy of the data MS to retain all of the important non-visibility data, and then use the [CASAtools table tools](https://casadocs.readthedocs.io/en/stable/api/tt/casatools.table.html#table) (such as tb.putcol()to insert the residual visibilities). 
- If you only want to read out of an MS and not put data back in, I recommend instead reading the methods available with [CASAtools ms tools](https://casadocs.readthedocs.io/en/stable/api/tt/casatools.ms.html#casatools.ms). 
- Please note that working with table tools and ms tools is tricky as the docs do provide some information on how the data is organized in the output arrays. (For example, the ms.get_data(items='amplitude') function returns data in the shape [correlations, channels, rows] and ms.get_data(items='amplitude', ifraxis=True) would return data in the shape [correlations, channels, baselines, rows]. But this is still a bit of a black box in terms of how explicitly the *rows* are sorted. When using something like [dask-ms](https://dask-ms.readthedocs.io/en/latest/) or [xarray-ms](https://xarray-ms.readthedocs.io/en/latest/), the sorting of these tables is labelled more explicitly, but I ended up just using a combination of uvplot and CASAtools in the end as my small data size allowed it, and I just accepted a little bit of uncertainty on the organization) 

*MCMC samplers*
- The [emcee docs](https://emcee.readthedocs.io/en/stable/) are very useful, especially the section on parallelization. This was something I struggled with and still remains questionably resolved in the code here, but this provided a good reference for troubleshooting. 
