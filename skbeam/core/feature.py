# ######################################################################
# Copyright (c) 2014, Brookhaven Science Associates, Brookhaven        #
# National Laboratory. All rights reserved.                            #
#                                                                      #
# Redistribution and use in source and binary forms, with or without   #
# modification, are permitted provided that the following conditions   #
# are met:                                                             #
#                                                                      #
# * Redistributions of source code must retain the above copyright     #
#   notice, this list of conditions and the following disclaimer.      #
#                                                                      #
# * Redistributions in binary form must reproduce the above copyright  #
#   notice this list of conditions and the following disclaimer in     #
#   the documentation and/or other materials provided with the         #
#   distribution.                                                      #
#                                                                      #
# * Neither the name of the Brookhaven Science Associates, Brookhaven  #
#   National Laboratory nor the names of its contributors may be used  #
#   to endorse or promote products derived from this software without  #
#   specific prior written permission.                                 #
#                                                                      #
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS  #
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT    #
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS    #
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE       #
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,           #
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES   #
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR   #
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)   #
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,  #
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OTHERWISE) ARISING   #
# IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE   #
# POSSIBILITY OF SUCH DAMAGE.                                          #
########################################################################
"""
This module contains code for extracting features from data
"""
from __future__ import absolute_import, division, print_function
from six.moves import zip
import numpy as np
from collections import deque
from .fitting import fit_quad_to_peak
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from skimage import feature, transform
import logging
logger = logging.getLogger(__name__)


class PeakRejection(Exception):
    """Custom exception class to indicate that the refine function rejected
    the candidate peak.

    This uses the exception handling framework in a method akin to
    `StopIteration` to indicate that there will be no return value.
    """
    pass


def peak_refinement(x, y, cands, window, refine_function, refine_args=None):
    """Refine candidate locations

    Parameters
    ----------
    x : array
        The independent variable, does not need to be evenly spaced.

    y : array
        The dependent variable.  Must correspond 1:1 with the values in `x`

    cands : array
        Array of the indices in `x` (and `y`) for the candidate peaks.

    refine_function : function
        A function which takes a section of data with a peak in it and returns
        the location and height of the peak to sub-sample accuracy.  Additional
        parameters can be passed through via the refine_args kwarg.
        The function signature must be::

            center, height = refine_func(x, y, **kwargs)

        This function may raise `PeakRejection` to indicate no suitable
        peak was found

    window : int
        How many samples to extract on either side of the
        candidate locations are passed to the refine function.  The
        window will be truncated near the boundaries.  The length of the
        data passed to the refine function will be (2 * window + 1).

    refine_args : dict, optional
        The passed to the refine_function

    Returns
    -------
    peak_locations : array
        The locations of the peaks

    peak_heights : array
        The heights of the peaks

    Examples
    --------
    >>> x = np.arange(512)
    >>> tt = np.zeros(512)
    >>> tt += np.exp(-((x - 150.55)/10)**2)
    >>> tt += np.exp(-((x - 450.75)/10)**2)
    >>> cands = scipy.signal.argrelmax(tt)[0]

    >>> print(peak_refinement(x, tt, cands, 10, refine_quadratic))
    (array([ 150.62286432,  450.7909412 ]), array([ 0.96435832,  0.96491501]))
    >>> print(peak_refinement(x, tt, cands, 10, refine_log_quadratic))
    (array([ 150.55,  450.75]), array([ 1.,  1.]))
    """
    # clean up input
    x = np.asarray(x)
    y = np.asarray(y)
    cands = np.asarray(cands, dtype=int)
    window = int(window)
    if refine_args is None:
        refine_args = dict()
    # local working variables
    out_tmp = deque()
    max_ind = len(x)

    for ind in cands:
        slc = slice(np.max([0, ind-window]),
                    np.min([max_ind, ind + window + 1]))
        try:
            ret = refine_function(x[slc], y[slc], **refine_args)
        except PeakRejection:
            # We are catching the PeakRejections raised here as
            # an indication that no suitable peak was found
            continue
        else:
            out_tmp.append(ret)

    return tuple([np.array(_) for _ in zip(*out_tmp)])


def refine_quadratic(x, y, Rval_thresh=None):
    """
    Attempts to refine the peaks by fitting to
    a quadratic function.

    Parameters
    ----------
    x : array
        Independent variable

    y : array
        Dependent variable

    Rval_thresh : float, optional
        Threshold for R^2 value of fit,  If the computed R^2 is worse than
        this threshold PeakRejection will be raised

    Returns
    -------
    center : float
        Refined estimate for center

    height : float
        Refined estimate for height

    Raises
    ------
    PeakRejection
       Raised to indicate that no suitable peak was found in the
       interval

    """
    beta, R2 = fit_quad_to_peak(x, y)
    if Rval_thresh is not None and R2 < Rval_thresh:
        raise PeakRejection()

    return beta[1], beta[2]


def refine_log_quadratic(x, y, Rval_thresh=None):
    """
    Attempts to refine the peaks by fitting a quadratic to the log of
    the y-data.  This is a linear approximation of fitting a Gaussian.

    Parameters
    ----------
    x : array
        Independent variable

    y : array
        Dependent variable

    Rval_thresh : float, optional
        Threshold for R^2 value of fit,  If the computed R^2 is worse than
        this threshold PeakRejection will be raised

    Returns
    -------
    center : float
        Refined estimate for center

    height : float
        Refined estimate for height

    Raises
    ------
    PeakRejection
       Raised to indicate that no suitable peak was found in the
       interval

    """
    beta, R2 = fit_quad_to_peak(x, np.log(y))
    if Rval_thresh is not None and R2 < Rval_thresh:
        raise PeakRejection()

    return beta[1], np.exp(beta[2])


def filter_n_largest(y, cands, N):
    """Filters the N largest candidate peaks

    Return a maximum of N largest candidates.  If N > len(cands) then
    all of the cands will be returned sorted, else the indices
    of the N largest peaks will be returned in descending order.

    Parameters
    ----------
    y : array
        Independent variable

    cands : array
        An array containing the indices of candidate peaks

    N : int
        The maximum number of peaks to return, sorted by size.
        Must be positive

    Returns
    -------
    cands : array
        An array of the indices of up to the N largest candidates
    """
    cands = np.asarray(cands)
    N = int(N)
    if N <= 0:
        raise ValueError("The maximum number of peaks to return must "
                         "be positive not {}".format(N))

    sorted_args = np.argsort(y[cands])
    # cut out if asking for more peaks than exist
    if len(cands) < N:
        return cands[sorted_args][::-1]

    return cands[sorted_args[-N:]][::-1]


def filter_peak_height(y, cands, thresh, window=5):
    """
    Filter to remove candidate that are too small.  This
    is implemented by looking at the relative height (max - min)
    of the peak in a window around the candidate peak.


    Parameters
    ----------
    y : array
        Independent variable

    cands : array
        An array containing the indices of candidate peaks

    thresh : int
        The minimum peak-to-peak size of the candidate peak to be accepted

    window : int, optional
        The size of the window around the peak to consider

    Returns
    -------
    cands : array
        An array of the indices which pass the filter

    """
    y = np.asarray(y)
    out_tmp = deque()
    max_ind = len(y)
    for ind in cands:
        slc = slice(np.max([0, ind-window]),
                    np.min([max_ind, ind + window + 1]))
        pk_hght = np.ptp(y[slc])
        if pk_hght > thresh:
            out_tmp.append(ind)

    return np.array(out_tmp)


# add our refinement functions as an attribute on peak_refinement
# ta make auto-wrapping for vistrials easier.
peak_refinement.refine_function = [refine_log_quadratic, refine_quadratic]

def find_center_rings_hough(image, num_rings, rad_range, intensity_thresh, sigma):
    """
    Take angled cut or multiple incremented cuts of image data.

    Parameters
    ----------
    image : ndarray
        Image data
    num_rings : int
        Number of candidate circles to be searched for
    rad_range : array
        The min, max, and increment for range of candidate radii to be searched
    intensity_thresh : array
        Min and max of intensity to be used in canny filter
    sigma : float
        Standard deviation applied in canny filter
    v_range : array
        Min and max for matplotlib colormap

    Returns
    -------
    center :list
        Center coordinates of candidate circles as (x,y) tuples
    radii: array
        radii of the candidate circles

    """
    low_thresh, hi_thresh = intensity_thresh
    sigma = sigma
    edges = feature.canny(image, sigma, low_threshold= low_thresh, high_threshold= hi_thresh)

    # Range of possible radii
    hough_radii = np.arange(*rad_range)
    hough_res = transform.hough_circle(edges, hough_radii)

    # Selects the most prominent circles, up to num_rings
    accums, cx, cy, radii = transform.hough_circle_peaks(hough_res, hough_radii,
                                               total_num_peaks=num_rings)

    centers = [(cx[i], cy[i]) for i in range(len(radii))]
    return centers, radii

def draw_rings(image, centers, radii, v_range = None):
    """
    Draw rings on an image with matplotlib

    Parameters
    ----------
    image : ndarray
        Image data
    centers : int
        center points of each circle to be drawn
    radii : array
        Radii of circles to be drawn
    v_range : array
        Min and max for matplotlib colormap

    """
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 4))

    for i in range(len(radii)):
        ax.scatter(centers[i][0], centers[i][1], s =20, c = 'red')
        circ = mpatches.Circle((centers[i][0], centers[i][1]), radii[i], facecolor='none', edgecolor='r')
        ax.add_artist(circ)

    if v_range != None:
        ax.imshow(image, cmap="viridis", vmin = v_range[0], vmax = v_range[1])
    else:
        ax.imshow(image, cmap="viridis")
    plt.show()
