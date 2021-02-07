#! encoding: utf-8
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
This module is for the 'core' data types.
"""
from __future__ import absolute_import, division, print_function
import six

from six.moves import zip
from six import string_types

import time
import sys

from collections import namedtuple, defaultdict, deque
from collections.abc import MutableMapping
import numpy as np
from itertools import tee

import logging
import scipy.stats as sts

from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)

md_value = namedtuple("md_value", ['value', 'units'])

_defaults = {
    "bins": 100,
    'nx': 100,
    'ny': 100,
    'nz': 100
}


class NotInstalledError(ImportError):
    '''
    Custom exception that should be subclassed to handle
    specific missing libraries

    '''
    pass


class MD_dict(MutableMapping):
    """
    A class to make dealing with the meta-data scheme for DataExchange easier

    Examples
    --------
    Getting and setting data by path is possible

    >>> tt = MD_dict()
    >>> tt['name'] = 'test'
    >>> tt['nested.a'] = 2
    >>> tt['nested.b'] = (5, 'm')
    >>> tt['nested.a'].value
    2
    >>> tt['nested.a'].units is None
    True
    >>> tt['name'].value
    'test'
    >>> tt['nested.b'].units
    'm'
    """

    def __init__(self, md_dict=None):
        # TODO properly walk the input on upgrade dicts -> MD_dict
        if md_dict is None:
            md_dict = dict()

        self._dict = md_dict
        self._split = '.'

    def __repr__(self):
        return self._dict.__repr__()

    # overload __setitem__ so dotted paths work
    def __setitem__(self, key, val):

        key_split = key.split(self._split)
        tmp = self._dict
        for k in key_split[:-1]:
            try:
                tmp = tmp[k]._dict
            except Exception:
                tmp[k] = type(self)()
                tmp = tmp[k]._dict
            if isinstance(tmp, md_value):
                # TODO make message better
                raise KeyError("trying to use a leaf node as a branch")

        # if passed in an md_value, set it and return
        if isinstance(val, md_value):
            tmp[key_split[-1]] = val
            return
        # catch the case of a bare string
        elif isinstance(val, string_types):
            # a value with out units
            tmp[key_split[-1]] = md_value(val, 'text')
            return
        # not something easy, try to guess what to do instead
        try:
            # if the second element is a string or None, cast to named tuple
            if isinstance(val[1], string_types) or val[1] is None:
                print('here')
                tmp[key_split[-1]] = md_value(*val)
            # else, assume whole thing is the value with no units
            else:
                tmp[key_split[-1]] = md_value(val, None)
        # catch any type errors from trying to index into non-indexable things
        # or from trying to use iterables longer than 2
        except TypeError:
            tmp[key_split[-1]] = md_value(val, None)

    def __getitem__(self, key):
        key_split = key.split(self._split)
        tmp = self._dict
        for k in key_split[:-1]:
            try:
                tmp = tmp[k]._dict
            except Exception:
                tmp[k] = type(self)()
                tmp = tmp[k]._dict

            if isinstance(tmp, md_value):
                # TODO make message better
                raise KeyError("trying to use a leaf node as a branch")

        return tmp.get(key_split[-1], None)

    def __delitem__(self, key):
        # pass one delete the entry
        # TODO make robust to non-keys
        key_split = key.split(self._split)
        tmp = self._dict
        for k in key_split[:-1]:
            # make sure we are grabbing the internal dict
            tmp = tmp[k]._dict
        del tmp[key_split[-1]]
        # TODO pass 2 remove empty branches

    def __len__(self):
        return len(list(iter(self)))

    def __iter__(self):
        return _iter_helper([], self._split, self._dict)


def _iter_helper(path_list, split, md_dict):
    """
    Recursively walk the tree and return the names of the leaves
    """
    for k, v in six.iteritems(md_dict):
        if isinstance(v, md_value):
            yield split.join(path_list + [k])
        else:
            for inner_v in _iter_helper(path_list + [k], split, v._dict):
                yield inner_v


class verbosedict(dict):
    """
    A sub-class of dict which raises more verbose errors if
    a key is not found.
    """

    def __getitem__(self, key):
        try:
            v = dict.__getitem__(self, key)
        except KeyError:
            if len(self) < 25:
                new_msg = ("You tried to access the key '{key}' "
                           "which does not exist.  The "
                           "extant keys are: {valid_keys}").format(
                    key=key, valid_keys=list(self))
            else:
                new_msg = ("You tried to access the key '{key}' "
                           "which does not exist.  There "
                           "are {num} extant keys, which is too many to "
                           "show you").format(
                    key=key, num=len(self))
            six.reraise(KeyError, KeyError(new_msg), sys.exc_info()[2])
        return v


class RCParamDict(MutableMapping):
    """A class to make dealing with storing default values easier.

    RC params is a hold- over from the UNIX days where configuration
    files are 'rc' files.  See
    http://en.wikipedia.org/wiki/Configuration_file

    Examples
    --------
    Getting and setting data by path is possible

    >>> tt = RCParamDict()
    >>> tt['name'] = 'test'
    >>> tt['nested.a'] = 2
    """
    _delim = '.'

    def __init__(self):
        # the dict to hold the keys at this level
        self._dict = dict()
        # the defaultdict (defaults to just accepting it) of
        # validator functions
        self._validators = defaultdict(lambda: lambda x: True)

    # overload __setitem__ so dotted paths work
    def __setitem__(self, key, val):
        # try to split the key
        splt_key = key.split(self._delim, 1)
        # if more than one part, recurse
        if len(splt_key) > 1:
            try:
                tmp = self._dict[splt_key[0]]
            except KeyError:
                tmp = RCParamDict()
                self._dict[splt_key[0]] = tmp

            if not isinstance(tmp, RCParamDict):
                raise KeyError("name space is borked")

            tmp[splt_key[1]] = val
        else:
            if not self._validators[key]:
                # TODO improve the validation error
                raise ValueError("fails to validate, improve this")
            self._dict[key] = val

    def __getitem__(self, key):
        # try to split the key
        splt_key = key.split(self._delim, 1)
        if len(splt_key) > 1:
            return self._dict[splt_key[0]][splt_key[1]]
        else:
            return self._dict[key]

    def __delitem__(self, key):
        splt_key = key.split(self._delim, 1)
        if len(splt_key) > 1:
            self._dict[splt_key[0]].__delitem__(splt_key[1])
        else:
            del self._dict[key]

    def __len__(self):
        return len(list(iter(self)))

    def __iter__(self):
        return self._iter_helper([])

    def _iter_helper(self, path_list):
        """
        Recursively walk the tree and return the names of the leaves
        """
        for key, val in six.iteritems(self._dict):
            if isinstance(val, RCParamDict):
                for k in val._iter_helper(path_list + [key, ]):
                    yield k
            else:
                yield self._delim.join(path_list + [key, ])

    def __repr__(self):
        # recursively get the formatted list of strings
        str_list = self._repr_helper(0)
        # return as a single string
        return '\n'.join(str_list)

    def _repr_helper(self, tab_level):
        # to accumulate the strings into
        str_list = []
        # list of the elements at this level
        elm_list = []
        # list of sub-levels
        nested_list = []
        # loop over the local _dict and sort out which
        # keys are nested and which are this level
        for key, val in six.iteritems(self._dict):
            if isinstance(val, RCParamDict):
                nested_list.append(key)
            else:
                elm_list.append(key)

        # sort the keys in both lists
        elm_list.sort()
        nested_list.sort()

        # loop over and format the keys/vals at this level
        for elm in elm_list:
            str_list.append("    " * tab_level +
                            "{key}: {val}".format(
                                key=elm, val=self._dict[elm]))
        # deal with the nested groups
        for nested in nested_list:
            # add the label for the group name
            str_list.append("    " * tab_level +
                            "{key}:".format(key=nested))
            # add the strings from _all_ the nested groups
            str_list.extend(
                self._dict[nested]._repr_helper(tab_level + 1))
        return str_list


keys_core = {
    "pixel_size": {
        "description": ("2 element tuple defining the (x y) dimensions of the "
                        "pixel"),
        "type": tuple,
        "units": "um",
    },
    "voxel_size": {
        "description": ("3 element tuple defining the (x y z) dimensions "
                        "of the voxel"),
        "type": tuple,
        "units": "um"
    },
    "calibrated_center": {
        "description": ("2 element tuple defining the (x y) center of the "
                        "detector in pixels"),
        "type": tuple,
        "units": "pixel",
    },
    "detector_size": {
        "description": ("2 element tuple defining no. of pixels(size) in the "
                        "detector X and Y direction"),
        "type": tuple,
        "units": "pixel",
    },
    "detector_tilt_angles": {
        "description": "Detector tilt angle",
        "type": tuple,
        "units": " degrees",
    },
    "dist_sample": {
        "description": "distance from the sample to the detector (mm)",
        "type": float,
        "units": "mm",
    },
    "wavelength": {
        "description": "wavelength of incident radiation (Angstroms)",
        "type": float,
        "units": "angstrom",
    },
    "ub_mat": {
        "description": "UB matrix(orientation matrix) 3x3 array",
        "type": "ndarray",
    },
    "energy": {
        "description": "scanning energy for data collection",
        "type": float,
        "units": "keV",
    },
    "array_dimensions": {
        "description": "axial lengths of the array (Pixels)",
        "x_dimension": {
            "description": "x-axis array length as int",
            "type": int,
            "units": "pixels"
        },
        "y_dimension": {
            "description": "y-axis array length as int",
            "type": int,
            "units": "pixels"
        },
        "z_dimension": {
            "description": "z-axis array length as int",
            "type": int,
            "units": "pixels"
        }
    },
    "bounding_box": {
        "description": ("physical extents of the array: useful for ",
                        "volume alignment, transformation, merge and ",
                        "spatial comparison of multiple volumes"),
        "x_min": {
            "description": "minimum spatial coordinate along the x-axis",
            "type": float,
            "units": "um"
        },
        "x_max": {
            "description": "maximum spatial coordinate along the x-axis",
            "type": float,
            "units": "um"
        },
        "y_min": {
            "description": "minimum spatial coordinate along the y-axis",
            "type": float,
            "units": "um"
        },
        "y_max": {
            "description": "maximum spatial coordinate along the y-axis",
            "type": float,
            "units": "um"
        },
        "z_min": {
            "description": "minimum spatial coordinate along the z-axis",
            "type": float,
            "units": "um"
        },
        "z_max": {
            "description": "maximum spatial coordinate along the z-axis",
            "type": float,
            "units": "um"
        },
    },
}


def subtract_reference_images(imgs, is_reference):
    """
    Function to subtract a series of measured images from
    background/dark current/reference images.  The nearest reference
    image in the reverse temporal direction is subtracted from each
    measured image.

    Parameters
    ----------
    imgs : numpy.ndarray
        Array of 2-D images

    is_reference : 1-D boolean array
        true  : image is reference image
        false : image is measured image

    Returns
    -------
    img_corr : numpy.ndarray
        len(img_corr) == len(img_arr) - len(is_reference_img == true)
        img_corr is the array of measured images minus the reference
        images.

    Raises
    ------
    ValueError
        Possible causes:
            is_reference contains no true values
            Raised when the first image in the array is not a reference image.

    """
    # an array of 1, 0, 1,.. should work too
    if not is_reference[0]:
        # use ValueError because the user passed in invalid data
        raise ValueError("The first image is not a reference image")
    # grab the first image
    ref_imge = imgs[0]
    # make an array of zeros of the correct type
    corrected_image = deque()
    # zip together (lazy like this is really izip), images and flags
    for imgs, ref in zip(imgs[1:], is_reference[1:]):
        # if this is a ref image, save it and move on
        if ref:
            ref_imge = imgs
            continue
        # else, do the subtraction
        corrected_image.append(imgs - ref_imge)

    # return the output as a list
    return list(corrected_image)


def img_to_relative_xyi(img, cx, cy, pixel_size_x=None, pixel_size_y=None):
    """
    Convert the 2D image to a list of x y I coordinates where
    x == x_img - detector_center[0] and
    y == y_img - detector_center[1]

    Parameters
    ----------
    img: `ndarray`
        2D image
    cx : float
        Image center in the x direction
    cy : float
        Image center in the y direction
    pixel_size_x : float, optional
        Pixel size in x
    pixel_size_y : float, optional
        Pixel size in y
    **kwargs: dict
        Bucket for extra parameters in an unpacked dictionary

    Returns
    -------
    x : `ndarray`
        x-coordinate of pixel. shape (N, )
    y : `ndarray`
        y-coordinate of pixel. shape (N, )
    I : `ndarray`
        intensity of pixel. shape (N, )
    """
    if pixel_size_x is not None and pixel_size_y is not None:
        if pixel_size_x <= 0:
            raise ValueError('Input parameter pixel_size_x must be greater '
                             'than 0. Your value was ' +
                             six.text_type(pixel_size_x))
        if pixel_size_y <= 0:
            raise ValueError('Input parameter pixel_size_y must be greater '
                             'than 0. Your value was ' +
                             six.text_type(pixel_size_y))
    elif pixel_size_x is None and pixel_size_y is None:
        pixel_size_x = 1
        pixel_size_y = 1
    else:
        raise ValueError('pixel_size_x and pixel_size_y must both be None or '
                         'greater than zero. You passed in values for '
                         'pixel_size_x of {0} and pixel_size_y of {1}'
                         ''.format(pixel_size_x, pixel_size_y))

    # Caswell's incredible terse rewrite
    x, y = np.meshgrid(pixel_size_x * (np.arange(img.shape[0]) - cx),
                       pixel_size_y * (np.arange(img.shape[1]) - cy))

    # return x, y and intensity as 1D arrays
    return x.ravel(), y.ravel(), img.ravel()


def bin_1D(x, y, nx=None, min_x=None, max_x=None):
    """
    Bin the values in y based on their x-coordinates

    Parameters
    ----------
    x : array
        position
    y : array
        intensity
    nx : integer, optional
        number of bins to use defaults to default bin value
    min_x : float, optional
        Left edge of first bin defaults to minimum value of x
    max_x : float, optional
        Right edge of last bin defaults to maximum value of x

    Returns
    -------
    edges : array
        edges of bins, length nx + 1

    val : array
        sum of values in each bin, length nx

    count : array
        The number of counts in each bin, length nx
    """

    # handle default values
    if min_x is None:
        min_x = np.min(x)
    if max_x is None:
        max_x = np.max(x)
    if nx is None:
        nx = int(max_x - min_x)

    # use a weighted histogram to get the bin sum
    bins = np.linspace(start=min_x, stop=max_x, num=nx + 1, endpoint=True)
    val, _ = np.histogram(a=x, bins=bins, weights=y)
    # use an un-weighted histogram to get the counts
    count, _ = np.histogram(a=x, bins=bins)
    # return the three arrays
    return bins, val, count


def radial_grid(center, shape, pixel_size=None):
    """Convert a cartesian grid (x,y) to the radius relative to some center

    Parameters
    ----------
    center : tuple
        point in image where r=0; may be a float giving subpixel precision.
        Order is (rr, cc).
    shape : tuple
        Image shape which is used to determine the maximum extent of output
        pixel coordinates.
        Order is (rr, cc).
    pixel_size : sequence, optional
        The physical size of the pixels.
        len(pixel_size) should be the same as len(shape)
        defaults to (1,1)

    Returns
    -------
    r : array
        The distance of each pixel from `center`
        Shape of the return value is equal to the `shape` input parameter
    """

    if pixel_size is None:
        pixel_size = (1, 1)

    X, Y = np.meshgrid(pixel_size[1] * (np.arange(shape[1]) - center[1]),
                       pixel_size[0] * (np.arange(shape[0]) - center[0]))
    return np.sqrt(X * X + Y * Y)


def angle_grid(center, shape, pixel_size=None):
    """
    Make a grid of angular positions.

    Read note for our conventions here -- there be dragons!

    Parameters
    ----------
    center : tuple
        point in image where r=0; may be a float giving subpixel precision.
        Order is (rr, cc).

    shape: tuple
        Image shape which is used to determine the maximum extent of output
        pixel coordinates. Order is (rr, cc).

    Returns
    -------
    agrid : array
        angular position (in radians) of each array element in range [-pi, pi]

    Note
    ----
    :math:`\\theta`, the counter-clockwise angle from the positive x axis,
    assuming the positive y-axis points upward.
    :math:`\\theta \\el [-\\pi, \\pi]`.  In array indexing and the conventional
    axes for images (origin in upper left), positive y is downward.
    """

    if pixel_size is None:
        pixel_size = (1, 1)

    # row is y, column is x. "so say we all. amen."
    x, y = np.meshgrid(pixel_size[1] * (np.arange(shape[1]) -
                                        center[1]),
                       pixel_size[0] * (np.arange(shape[0]) -
                                        center[0]))
    return np.arctan2(y, x)


def radius_to_twotheta(dist_sample, radius):
    """
    Converts radius from the calibrated center to scattering angle
    (2:math:`2\\theta`) with known detector to sample distance.

    Parameters
    ----------
    dist_sample : float
        distance from the sample to the detector (mm)

    radius : array
        The L2 norm of the distance of each pixel from the calibrated center.

    Returns
    -------
    two_theta : array
        An array of :math:`2\\theta` values
    """
    return np.arctan(radius / dist_sample)


def wedge_integration(src_data, center, theta_start,
                      delta_theta, r_inner, delta_r):
    """
    Implementation of caking.

    Parameters
    ----------
    scr_data : ndarray
        The source-data to be integrated

    center : ndarray
        The center of the ring in pixels

    theta_start : float
        The angle of the start of the wedge from the
        image y-axis in degrees

    delta_theta : float
        The angular width of the wedge in degrees.  Positive
        angles go clockwise, negative go counter-clockwise.

    r_inner : float
        The inner radius in pixel units, Must be non-negative

    delta_r : float
        The length of the wedge in the radial direction
        in pixel units. Must be non-negative

    Returns
    -------
    float
        The integrated intensity under the wedge
    """
    raise NotImplementedError()


def bin_edges(range_min=None, range_max=None, nbins=None, step=None):
    """
    Generate bin edges.  The last value in the returned array is
    the right edge of the last bin, the rest of the values are the
    left edges of each bin.

    If `range_max` is specified all bin edges will be less than or
    equal to it's value.

    If `range_min` is specified all bin edges will be greater than
    or equal to it's value

    If `nbins` is specified then there will be than number of bins and
    the returned array will have length `nbins + 1` (as the right most
    edge is included)

    If `step` is specified then bin width is approximately `step` (It is
    not exact due to the nature of floats). The arrays generated by
    `np.cumsum(np.ones(nbins) * step)` and `np.arange(nbins) * step` are
    not identical.  This function uses the second method in all cases
    where `step` is specified.

    .. warning :: If the set :code:`(range_min, range_max, step)` is
        given there is no guarantee that :code:`range_max - range_min`
        is an integer multiple of :code:`step`.  In this case the left
        most bin edge is :code:`range_min` and the right most bin edge
        is less than :code:`range_max` and the distance between the
        right most edge and :code:`range_max` is not greater than
        :code:`step` (this is the same behavior as the built-in
        :code:`range()`).  It is not recommended to specify bins in this
        manner.

    Parameters
    ----------
    range_min : float, optional
        The minimum value that may be included as a bin edge

    range_max : float, optional
        The maximum value that may be included as a bin edge

    nbins : int, optional
        The number of bins, if specified the length of the returned
        value will be nbins + 1

    step : float, optional
        The step between the bins

    Returns
    -------
    edges : np.array
        An array of floats for the bin edges.  The last value is the
        right edge of the last bin.
    """
    num_valid_args = sum((range_min is not None, range_max is not None,
                          step is not None, nbins is not None))
    if num_valid_args != 3:
        raise ValueError("Exactly three of the arguments must be non-None "
                         "not {}.".format(num_valid_args))

    if range_min is not None and range_max is not None:
        if range_max <= range_min:
            raise ValueError("The minimum must be less than the maximum")

    if nbins is not None:
        if nbins <= 0:
            raise ValueError("The number of bins must be positive")

    # The easy case
    if step is None:
        return np.linspace(range_min, range_max, nbins + 1, endpoint=True)

    # in this case, the user gave use min, max, and step
    if nbins is None:
        if step > (range_max - range_min):
            raise ValueError("The step can not be greater than the difference "
                             "between min and max")
        nbins = int((range_max - range_min) // step)
        ret = range_min + np.arange(nbins + 1) * step
        # if the last value is greater than the max (should never happen)
        if ret[-1] > range_max:
            return ret[:-1]
        if range_max - ret[-1] > 1e-10 * step:
            logger.debug("Inconsistent "
                         "(range_min, range_max, step) "
                         "and step does not evenly divide "
                         "(range_min - range_max). "
                         "The bins has been truncated.\n"
                         "min: %f max: %f step: %f gap: %f",
                         range_min, range_max,
                         step, range_max - ret[-1])
        return ret

    # in this case we got range_min, nbins, step
    if range_max is None:
        return range_min + np.arange(nbins + 1) * step

    # in this case we got range_max, nbins, step
    if range_min is None:
        return range_max - np.arange(nbins + 1)[::-1] * step


def grid3d(q, img_stack,
           nx=None, ny=None, nz=None,
           xmin=None, xmax=None, ymin=None,
           ymax=None, zmin=None, zmax=None,
           binary_mask=None):
    """Grid irregularly spaced data points onto a regular grid via histogramming

    This function will process the set of reciprocal space values (q), the
    image stack (img_stack) and grid the image data based on the bounds
    provided, using defaults if none are provided.

    Parameters
    ----------
    q : ndarray
        (Qx, Qy, Qz) - HKL values - Nx3 array
    img_stack : ndarray
        Intensity array of the images
        dimensions are: [num_img][num_rows][num_cols]
    nx : int, optional
        Number of voxels along x
    ny : int, optional
        Number of voxels along y
    nz : int, optional
        Number of voxels along z
    xmin : float, optional
        Minimum value along x. Defaults to smallest x value in q
    ymin : float, optional
        Minimum value along y. Defaults to smallest y value in q
    zmin : float, optional
        Minimum value along z. Defaults to smallest z value in q
    xmax : float, optional
        Maximum value along x. Defaults to largest x value in q
    ymax : float, optional
        Maximum value along y. Defaults to largest y value in q
    zmax : float, optional
        Maximum value along z. Defaults to largest z value in q
    binary_mask : ndarray, optional
        The binary mask provides a mechanism to remove unwanted pixels
        from the images.
        Binary mask can be two different shapes.
        - 1: 2-D with binary_mask.shape == np.asarray(img_stack[0]).shape
        - 2: 3-D with binary_mask.shape == np.asarray(img_stack).shape

    Returns
    -------
    mean : ndarray
        intensity grid.  The values in this grid are the
        mean of the values that fill with in the grid.
    occupancy : ndarray
        The number of data points that fell in the grid.
    std_err : ndarray
        This is the standard error of the value in the
        grid box.
    bounds : list
        tuple of (min, max, step) for x, y, z in order: [x_bounds,
        y_bounds, z_bounds]

    """
    try:
        from ..ext import ctrans
    except ImportError:
        raise NotImplementedError(
            "ctrans is not available on your platform. See "
            "https://github.com/scikit-beam/scikit-beam/issues/418 "
            "to follow updates to this problem.")

    # validate input
    img_stack = np.asarray(img_stack)
    # todo determine if we're going to support masked arrays
    # todo masked arrays seemed to have been punted to `process_to_q`

    # check to see if the binary mask and the image stack are identical shapes
    if binary_mask is None or binary_mask.shape == img_stack.shape:
        # do a dance :)
        pass
    elif binary_mask.shape == img_stack[0].shape:
        # this is still a valid mask, so make it the same dimensions
        # as img_stack.
        # should probably change this to use something similar to:
        # todo http://stackoverflow.com/questions/5564098/
        binary_mask = np.tile(np.ravel(binary_mask), img_stack.shape[0])

    else:
        raise ValueError("The binary mask must be the same shape as the"
                         "img_stack ({0}) or a single image in the image "
                         "stack ({1}).  The input binary mask is shaped ({2})"
                         "".format(img_stack.shape, img_stack[0].shape,
                                   binary_mask.shape))

    q = np.atleast_2d(q)
    if q.ndim != 2:
        raise ValueError("q.ndim must be a 2-D array of shape Nx3 array. "
                         "You provided an array with {0} dimensions."
                         "".format(q.ndim))
    if q.shape[1] != 3:
        raise ValueError("The shape of q must be an Nx3 array, not {0}X{1}"
                         " which you provided.".format(*q.shape))

    # set defaults for qmin, qmax, dq
    qmin = np.min(q, axis=0)
    qmax = np.max(q, axis=0)
    dqn = [_defaults['nx'], _defaults['ny'], _defaults['nz']]

    # pad the upper edge by just enough to ensure that all of the
    # points are in-bounds with the binning rules: lo <= val < hi
    qmax += np.spacing(qmax)

    # check for non-default input
    for target, input_vals in ((dqn, (nx, ny, nz)),
                               (qmin, (xmin, ymin, zmin)),
                               (qmax, (xmax, ymax, zmax))):
        for j, in_val in enumerate(input_vals):
            if in_val is not None:
                target[j] = in_val

    # format bounds
    bounds = np.array([qmin, qmax, dqn]).T

    # creating (Qx, Qy, Qz, I) Nx4 array - HKL values and Intensity
    # getting the intensity value for each pixel
    q = np.insert(q, 3, np.ravel(img_stack), axis=1)
    if binary_mask is not None:
        q = q[np.ravel(binary_mask)]

    # 3D grid of the data set
    # starting time for gridding
    t1 = time.time()

    # call the c library

    total, total2, occupancy, std_err = ctrans.grid3d(q, qmin, qmax, dqn)
    mean = total / occupancy

    # ending time for the gridding
    t2 = time.time()
    logger.info("Done processed in {0} seconds".format(t2 - t1))

    # No. of values zero in the grid
    empt_nb = (occupancy == 0).sum()

    # log some information about the grid at the debug level
    logger.debug("There are %2e bins in the grid", mean.size)
    if empt_nb:
        logger.debug("There are %.2e values zero in the grid", empt_nb)

    return mean, occupancy, std_err, bounds


def bin_edges_to_centers(input_edges):
    """
    Helper function for turning a array of bin edges into
    an array of bin centers

    Parameters
    ----------
    input_edges : array-like
        N + 1 values which are the left edges of N bins
        and the right edge of the last bin

    Returns
    -------
    centers : ndarray
        A length N array giving the centers of the bins
    """
    input_edges = np.asarray(input_edges)
    return (input_edges[:-1] + input_edges[1:]) * 0.5


# https://docs.python.org/2/library/itertools.html#recipes
def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def q_to_d(q):
    """
    Helper function to convert :math:`d` to :math:`q`.  The point
    of this function is to prevent fat-fingered typos.

    By definition the relationship is:

    .. math::

        q = \\frac{2 \\pi}{d}


    Parameters
    ----------
    q : array
        An array of q values

    Returns
    -------
    d : array
       An array of d (plane) spacing in the inverse of the units of ``q``

    """
    return (2 * np.pi) / np.asarray(q)


def d_to_q(d):
    """
    Helper function to convert :math:`d` to :math:`q`.
    The point of this function is to prevent fat-fingered typos.

    By definition the relationship is:

    .. math::

        d = \\frac{2 \\pi}{q}

    Parameters
    ----------
    d : array
       An array of d (plane) spacing

    Returns
    -------
    q : array
        An array of q values in the inverse of the units of ``d``


    """
    return (2 * np.pi) / np.asarray(d)


def q_to_twotheta(q, wavelength):
    r"""
    Helper function to convert q to two-theta.

    By definition the relationship is:

    .. math::

        \sin\left(\frac{2\theta}{2}\right) = \frac{\lambda q}{4 \pi}

    thus

    .. math::

        2\theta_n = 2 \arcsin\left(\frac{\lambda q}{4 \pi}\right)

    Parameters
    ----------
    q : array
        An array of :math:`q` values

    wavelength : float
        Wavelength of the incoming x-rays

    Returns
    -------
    two_theta : array
        An array of :math:`2\theta` values in radians
    """
    q = np.asarray(q)
    wavelength = float(wavelength)
    pre_factor = wavelength / (4 * np.pi)
    return 2 * np.arcsin(q * pre_factor)


def twotheta_to_q(two_theta, wavelength):
    r"""
    Helper function to convert two-theta to q

    By definition the relationship is

    .. math::

        \sin\left(\frac{2\theta}{2}\right) = \frac{\lambda q}{4 \pi}

    thus

    .. math::

        q = \frac{4 \pi \sin\left(\frac{2\theta}{2}\right)}{\lambda}



    Parameters
    ----------
    two_theta : array
        An array of :math:`2\theta` values

    wavelength : float
        Wavelength of the incoming x-rays

    Returns
    -------
    q : array
        An array of :math:`q` values in the inverse of the units
        of ``wavelength``
    """
    two_theta = np.asarray(two_theta)
    wavelength = float(wavelength)
    pre_factor = ((4 * np.pi) / wavelength)
    return pre_factor * np.sin(two_theta / 2)


def multi_tau_lags(multitau_levels, multitau_channels):
    """
    Standard multiple-tau algorithm for finding the lag times (delay
    times).

    Parameters
    ----------
    multitau_levels : int
        number of levels of multiple-taus
    multitau_channels : int
        number of channels or number of buffers in auto-correlators
        normalizations (must be even)

    Returns
    -------
    total_channels : int
        total number of channels ( or total number of delay times)
    lag_steps : ndarray
        delay or lag steps for the multiple tau analysis
    dict_lags : dict
        dictionary of delays for each multitau_levels

    Notes
    -----
    The multi-tau correlation scheme was used for finding the lag times
    (delay times).

    References: text [1]_

    .. [1] K. Schätzela, M. Drewela and  S. Stimaca, "Photon correlation
       measurements at large lag times: Improving statistical accuracy,"
       J. Mod. Opt., vol 35, p 711–718, 1988.
    """

    if (multitau_channels % 2 != 0):
        raise ValueError("Number of  multiple tau channels(buffers)"
                         " must be even. You provided {0} "
                         .format(multitau_channels))

    # total number of channels ( or total number of delay times)
    tot_channels = (multitau_levels + 1) * multitau_channels // 2

    lag = []
    dict_lags = {}
    lag_steps = np.arange(0, multitau_channels)
    dict_lags[1] = lag_steps
    for i in range(2, multitau_levels + 1):
        y = []
        for j in range(0, multitau_channels // 2):
            value = (multitau_channels // 2 + j) * (2 ** (i - 1))
            lag.append(value)
            y.append(value)
        dict_lags[i] = y

    lag_steps = np.append(lag_steps, np.array(lag))
    return tot_channels, lag_steps, dict_lags


def geometric_series(common_ratio, number_of_images, first_term=1):
    """
    This will provide the geometric series for the integration.
    Last values of the series has to be less than or equal to number
    of images
    ex: number_of_images = 100, first_term =1
    common_ratio = 2, geometric_series =  1, 2, 4, 8, 16, 32, 64
    common_ratio = 3, geometric_series =  1, 3, 9, 27, 81

    Parameters
    ----------
    common_ratio : float
        common ratio of the series

    number_of_images : int
        number of images

    first_term : float, optional
        first term in the series

    Return
    ------
    geometric_series : list
        time series

    Notes
    -----
    .. math::
        a + ar + ar^2 + ar^3 + ar^4 + ...

    a - first term in the series

    r - is the common ratio
    """

    geometric_series = [first_term]

    while geometric_series[-1] * common_ratio < number_of_images:
        geometric_series.append(geometric_series[-1] * common_ratio)
    return geometric_series


def bin_grid(image, r_array, pixel_sizes, statistic='mean', mask=None,
             bins=None):
    """
    Bin and integrate an image, given the radial array of pixels

    Parameters
    ----------
    image: np.array
        The image in quesion
    r_array: np.array
        The array which maps pixel positions to tilt/rotation corrected radii
    pixel_sizes: tuple
        The size of the pixels in the same units as the r_array
    statistic: str or func, optional
        The statistic to compute over the integration, defaults to mean
    mask: bool array, optional
        The array of pixels to be removed from the image before integration
    bins: array, optional
        The bins to use in the integration, if none given the function will
        give its best assessment based on the pixel_size and r_array

    Returns
    -------
    bin_centers : array
        The center of each bin in R
    int_stat : array
        Radial integrated statistic of the image.

    See Also
    --------
    circular_average : circularly average an image, assuming linear radial
        spacing (less general)

    """
    if mask is None:
        mask = np.ones(image.shape, dtype=int).astype(bool)
    if bins is None:
        res = np.hypot(*pixel_sizes)
        bins = np.arange(np.min(r_array) - res * .5,
                         np.max(r_array) + res * .5, res)

    int_stat, bin_edge, bin_num = sts.binned_statistic(r_array[mask],
                                                       image[mask],
                                                       statistic=statistic,
                                                       bins=bins)

    bin_centers = bin_edges_to_centers(bin_edge)

    return bin_centers, int_stat

    def angular_cut(image, angle, increment = None, stop = None,
                    mute_graph = False, v_min = None, v_max = None, num =1000):
        """
        Take angled cut or multiple incremented cuts of a image data.

        Parameters
        ----------
        image : ndarray
            Image data
        angle : float
            Desired angle for cut to be taken counter-clockwise from the horizontal axis
        increment : float
            The angular increment between line cuts if more than one is desired
        stop : float
            Maximum value for a set of incremented cuts
        mute_graph : boolean
            Mute the image and intensity plot
        v_min, v_max : float
            Set limits for matplotlib color map values
        num : float
            Number of intensity data points sampled between endpoints of angular cut

        Returns
        -------
        data : list
            A list of tuples that contains data from each cut. Each tuple
            contains the angle of the cut taken, an array of the intensity
            values sampled along the line cut and an array of the
            corresponding radii relative to center at each point sampled
            along the line cut.

        """
        height = image.shape[0]
        width = image.shape[1]
        center_x = (width - 1)/2.0
        center_y = (height - 1)/2.0
        if mute_graph == False:
            fig, axes = plt.subplots(nrows=2, figsize = (10,10))
            axes[0].scatter(center_x, center_y, s= 50, c='red')

        curr = angle
        data = []

        t = image

        while True:
            q = -1

            theta = curr*(np.pi)/180

            threshold = np.arctan(height/width)

            row0 = 0 #two endpoints of line at angle theta
            col0 = 0

            row1 = 0
            col1 = 0
            if theta >= np.pi:
                theta = theta - np.pi
            if theta > threshold or theta <= np.pi - threshold:
                row0 = 0
                row1 = height - 1
                col0 = int((width/2)+(height/2)*np.tan((np.pi/2) - theta))
                col1 = int((width/2)-(height/2)*np.tan((np.pi/2) - theta))
                if col1 >= height:
                    col1 -= 1
            if theta <= threshold:
                row0 = int((height/2) - (width/2)*np.tan(theta))
                row1 = int((height/2) + (width/2)*np.tan(theta))
                if row1 >= height:
                    row1 -= 1
                col0 = width - 1
                col1 = 0

            if theta > np.pi - threshold:
                row0 = int((height/2) + (width/2)*np.tan(theta))
                row1 = int((height/2) - (width/2)*np.tan(theta))
                col0 = 0
                col1 = width - 1


            r, c = np.linspace(row0, row1, num), np.linspace(col0, col1, num) #sample 1000 points between endpoints
            im_slice = t[r.astype(np.int), c.astype(np.int)]  #extract data from these indexed locations
            im_slice = im_slice[::q]
            radius = []
            grid = radial_grid(center = (center_y, center_x), shape = (height, width))

            for i in range(len(r)):
                p1 = int(r[i])
                p2 = int(c[i])
                rad = grid[p1][p2]
                radius.append(rad)

            radius = radius[::q]

            if mute_graph == False:
                if(v_min != None and v_max != None):
                    axes[0].imshow(t, vmin= v_min,vmax =  v_max)
                else:
                    axes[0].imshow(t)

                axes[0].plot([col0, col1], [row0, row1], 'ro-')
                axes[0].axis('image')

                axes[1].plot(radius, im_slice)
                plt.xlabel("Radius", fontsize = 15)
                plt.ylabel("Intensity", fontsize = 15)

            data.append((curr, im_slice, radius))

            if increment == None:
                break
            if curr + increment > stop:
                break
            curr = curr + increment

        if(mute_graph == False and center_x != None and center_y != None):
                    axes[0].scatter(center_x, center_y, s= 50, c='red')
        plt.show()
        return data
