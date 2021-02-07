"""
Copyright 2001, 2002 Enthought, Inc.
All rights reserved.

Copyright 2003-2013 SciPy Developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

- Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

- Redistributions in binary form must reproduce the above
  copyright notice, this list of conditions and the following
  disclaimer in the documentation and/or other materials provided
  with the distribution.

- Neither the name of Enthought nor the names of the SciPy Developers
  may be used to endorse or promote products derived from this software
  without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from __future__ import division, print_function, absolute_import

import warnings

import numpy as np
from ..utils import radial_grid, angle_grid, bin_edges_to_centers


class BinnedStatisticDD(object):
    std_ = ('mean', 'median', 'count', 'sum', 'std')

    def __init__(self, sample, statistic='mean',
                 bins=10, range=None, mask=None):
        """
        Compute a multidimensional binned statistic for a set of data.

        This is a generalization of a histogramdd function.  A
        histogram divides the space into bins, and returns the count
        of the number of points in each bin.  This function allows the
        computation of the sum, mean, median, or other statistic of
        the values within each bin.

        Parameters
        ----------
        sample : array_like
            Data to histogram passed as a sequence of D arrays of length N, or
            as an (N,D) array.
        statistic : string or callable, optional
            The statistic to compute (default is 'mean'). To compute multiple
            statistics efficiently, override this at __call__ time.
            The following statistics are available:

              * 'mean' : compute the mean of values for points within each bin.
                Empty bins will be represented by NaN.
              * 'median' : compute the median of values for points within each
                bin. Empty bins will be represented by NaN.
              * 'count' : compute the count of points within each bin.  This is
                identical to an unweighted histogram.  `values` array is not
                referenced.
              * 'sum' : compute the sum of values for points within each bin.
                This is identical to a weighted histogram.
              * function : a user-defined function which takes a 1D array of
                values, and outputs a single numerical statistic. This function
                will be called on the values in each bin.  Empty bins will be
                represented by function([]), or NaN if this returns an error.
        bins : sequence or int, optional
            The bin specification:

              * A sequence of arrays describing the bin edges along each
                dimension.
              * The number of bins for each dimension (nx, ny, ... =bins)
              * The number of bins for all dimensions (nx=ny=...=bins).
        range : sequence, optional
            A sequence of lower and upper bin edges to be used if the
            edges are not given explicitely in `bins`. Defaults to the
            minimum and maximum values along each dimension.
        mask : array_like
            array of ones and zeros with total size N (see documentation
            for `sample`). Values with mask==0 will be ignored.

        Note: If using numpy versions < 1.10.0, you may notice slow behavior of
        this constructor. This has to do with digitize, which was optimized
        from 1.10.0 onwards.
        """

        # This code is based on np.histogramdd
        try:
            # Sample is an ND-array.
            N, self.D = sample.shape
        except (AttributeError, ValueError):
            # Sample is a sequence of 1D arrays.
            sample = np.atleast_2d(sample).T
            N, self.D = sample.shape

        self.nbin = np.empty(self.D, int)
        self.edges = self.D * [None]
        self._centers = self.D * [None]
        dedges = self.D * [None]

        try:
            M = len(bins)
            if M != self.D:
                raise AttributeError('The dimension of bins must be equal '
                                     'to the dimension of the sample x.')
        except TypeError:
            bins = self.D * [bins]

        # Select range for each dimension
        # Used only if number of bins is given.
        if range is None:
            smin = np.atleast_1d(np.array(sample.min(0), float))
            smax = np.atleast_1d(np.array(sample.max(0), float))
        else:
            smin = np.zeros(self.D)
            smax = np.zeros(self.D)
            for i in np.arange(self.D):
                smin[i], smax[i] = range[i]

        # Make sure the bins have a finite width.
        for i in np.arange(len(smin)):
            if smin[i] == smax[i]:
                smin[i] = smin[i] - .5
                smax[i] = smax[i] + .5

        # Create edge arrays
        for i in np.arange(self.D):
            if np.isscalar(bins[i]):
                self.nbin[i] = bins[i] + 2  # +2 for outlier bins
                self.edges[i] = np.linspace(smin[i], smax[i], self.nbin[i] - 1)
            else:
                self.edges[i] = np.asarray(bins[i], float)
                self.nbin[i] = len(self.edges[i]) + 1  # +1 for outlier bins
            self._centers[i] = bin_edges_to_centers(self.edges[i])
            dedges[i] = np.diff(self.edges[i])

        self.nbin = np.asarray(self.nbin)

        # Compute the bin number each sample falls into.
        Ncount = {}
        for i in np.arange(self.D):
            # Apply mask in a non-ideal way by setting value outside range.
            # Would be better to do this using bincount "weights", perhaps.
            thissample = sample[:, i]
            if mask is not None:
                thissample[mask == 0] = (self.edges[i][0] -
                                         0.01 * (
                                             1 + np.fabs(self.edges[i][0])))
            Ncount[i] = np.digitize(thissample, self.edges[i])

        # Using digitize, values that fall on an edge are put in the
        # right bin.  For the rightmost bin, we want values equal to
        # the right edge to be counted in the last bin, and not as an
        # outlier.

        for i in np.arange(self.D):
            # Rounding precision
            decimal = int(-np.log10(dedges[i].min())) + 6
            # Find which points are on the rightmost edge.
            on_edge = np.where(np.around(sample[:, i], decimal) ==
                               np.around(self.edges[i][-1], decimal))[0]
            # Shift these points one bin to the left.
            Ncount[i][on_edge] -= 1

        # Compute the sample indices in the flattened statistic matrix.
        # TODO: These variables should be labeled as private, self._ni = ...
        #  We dont want to encourage the mutation of these parameters after
        #  instantiation.
        self.ni = self.nbin.argsort()
        self.xy = np.zeros(N, int)
        for i in np.arange(0, self.D - 1):
            self.xy += Ncount[self.ni[i]] * self.nbin[self.ni[i + 1:]].prod()
        self.xy += Ncount[self.ni[-1]]
        self._flatcount = None  # will be computed if needed
        self._argsort_index = None
        self.statistic = statistic

    @property
    def binmap(self):
        ''' Return the map of the bins per dimension.
                i.e. reverse transformation of flattened to unflattened bins

            Returns
            -------
            D np.ndarrays of length N where D is the number of dimensions
                and N is the number of data points.
                For each dimension, the min bin id is 0 and max n+1 where n is
                the number of bins in that dimension. The ids 0 and n+1 mark
                the outliers of the bins.
        '''
        N, = self.xy.shape
        binmap = np.zeros((self.D, N), dtype=int)
        denominator = 1

        for i in range(self.D):
            ind = self.D - i - 1
            subbinmap = (self.xy // denominator)
            if i < self.D - 1:
                subbinmap = subbinmap % self.nbin[self.ni[ind - 1]]
            binmap[ind] = subbinmap
            denominator *= self.nbin[self.ni[ind]]

        return binmap

    @property
    def flatcount(self):
        # Compute flatcount the first time it is accessed. Some statistics
        # never access it.
        if self._flatcount is None:
            self._flatcount = np.bincount(self.xy, None)
        return self._flatcount

    @property
    def argsort_index(self):
        # Compute argsort the first time it is accessed. Some statistics
        # never access it.
        if self._argsort_index is None:
            self._argsort_index = self.xy.argsort()
        return self._argsort_index

    @property
    def bin_edges(self):
        """
        bin_edges : array of dtype float
        Return the bin edges ``(length(statistic)+1)``.
        """
        return self.edges

    @property
    def bin_centers(self):
        """
        bin_centers : array of dtype float
        Return the bin centers ``(length(statistic))``.
        """
        return self._centers

    @property
    def statistic(self):
        return self._statistic

    @statistic.setter
    def statistic(self, new_statistic):
        if not callable(new_statistic) and new_statistic not in self.std_:
            raise ValueError('invalid statistic %r' % (new_statistic,))
        else:
            self._statistic = new_statistic

    def __call__(self, values, statistic=None):
        """
        Parameters
        ----------
        values : array_like
            The values on which the statistic will be computed.  This must be
            the same shape as `sample` in the constructor.
        statistic : string or callable, optional
            The statistic to compute (default is whatever was passed in when
            this object was instantiated).
            The following statistics are available:

              * 'mean' : compute the mean of values for points within each bin.
                Empty bins will be represented by NaN.
              * 'median' : compute the median of values for points within each
                bin. Empty bins will be represented by NaN.
              * 'count' : compute the count of points within each bin.  This is
                identical to an unweighted histogram.  `values` array is not
                referenced.
              * 'sum' : compute the sum of values for points within each bin.
                This is identical to a weighted histogram.
              * function : a user-defined function which takes a 1D array of
                values, and outputs a single numerical statistic. This function
                will be called on the values in each bin.  Empty bins will be
                represented by function([]), or NaN if this returns an error.

        Returns
        -------
        statistic_values : array
            The values of the selected statistic in each bin.
        """
        if statistic is None:
            statistic = self.statistic

        self.result = np.empty(self.nbin.prod(), float)
        if statistic == 'mean':
            self.result.fill(np.nan)
            flatsum = np.bincount(self.xy, values)
            a = self.flatcount.nonzero()
            self.result[a] = flatsum[a] / self.flatcount[a]
        elif statistic == 'std':
            self.result.fill(0)
            flatsum = np.bincount(self.xy, values)
            flatsum2 = np.bincount(self.xy, values ** 2)
            a = self.flatcount.nonzero()
            self.result[a] = np.sqrt(flatsum2[a] / self.flatcount[a] -
                                     (flatsum[a] / self.flatcount[a]) ** 2)
        elif statistic == 'count':
            self.result.fill(0)
            a = np.arange(len(self.flatcount))
            self.result[a] = self.flatcount
        elif statistic == 'sum':
            self.result.fill(0)
            flatsum = np.bincount(self.xy, values)
            a = np.arange(len(flatsum))
            self.result[a] = flatsum
        elif callable(statistic) or statistic == 'median':
            if statistic == 'median':
                internal_statistic = np.median
            else:
                internal_statistic = statistic
            with warnings.catch_warnings():
                # Numpy generates a warnings for mean/std/... with empty list
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                old = np.seterr(invalid='ignore')
                try:
                    null = internal_statistic([])
                except Exception:
                    null = np.nan
                np.seterr(**old)
            self.result.fill(null)

            vfs = values[self.argsort_index]
            i = 0
            for j, k in enumerate(self.flatcount):
                if k > 0:
                    self.result[j] = internal_statistic(vfs[i: i + k])
                i += k

        # Shape into a proper matrix
        self.result = self.result.reshape(np.sort(self.nbin))
        ni = np.copy(self.ni)
        for i in np.arange(self.nbin.size):
            j = ni.argsort()[i]
            self.result = self.result.swapaxes(i, j)
            ni[i], ni[j] = ni[j], ni[i]

        # Remove outliers (indices 0 and -1 for each dimension).
        core = self.D * [slice(1, -1)]
        self.result = self.result[tuple(core)]

        if (self.result.shape != self.nbin - 2).any():
            raise RuntimeError('Internal Shape Error')

        return self.result


class BinnedStatistic1D(BinnedStatisticDD):
    def __init__(self, x, statistic='mean',
                 bins=10, range=None, mask=None):
        """
        A refactored version of scipy.stats.binned_statistic to improve
        performance for the case where binning doesn't need to be
        re-initialized on every call.

        Compute a binned statistic for a set of data.

        This is a generalization of a histogram function.  A histogram divides
        the space into bins, and returns the count of the number of points in
        each bin.  This function allows the computation of the sum, mean,
        median, or other statistic of the values within each bin.

        Parameters
        ----------
        x : array_like
            A sequence of values to be binned.
        statistic : string or callable, optional
            The statistic to compute (default is 'mean').
            The following statistics are available:

              * 'mean' : compute the mean of values for points within each bin.
                Empty bins will be represented by NaN.
              * 'median' : compute the median of values for points within each
                bin. Empty bins will be represented by NaN.
              * 'count' : compute the count of points within each bin.  This is
                identical to an unweighted histogram.  `values` array is not
                referenced.
              * 'sum' : compute the sum of values for points within each bin.
                This is identical to a weighted histogram.
              * function : a user-defined function which takes a 1D array of
                values, and outputs a single numerical statistic. This function
                will be called on the values in each bin.  Empty bins will be
                represented by function([]), or NaN if this returns an error.
        bins : int or sequence of scalars, optional
            If `bins` is an int, it defines the number of equal-width bins in
            the given range (10 by default).  If `bins` is a sequence, it
            defines the bin edges, including the rightmost edge, allowing for
            non-uniform bin widths.  Values in `x` that are smaller than lowest
            bin edge are assigned to bin number 0, values beyond the highest
            bin are assigned to ``bins[-1]``.
        range : (float, float) or [(float, float)], optional
            The lower and upper range of the bins.  If not provided, range
            is simply ``(x.min(), x.max())``.  Values outside the range are
            ignored.
        mask : array_like
            ones and zeros with the same shape as `x`.
            Values with mask==0 will be ignored.

        See Also
        --------
        numpy.histogram, binned_statistic_2d, binned_statistic_dd

        Notes
        -----
        All but the last (righthand-most) bin is half-open.  In other words, if
        `bins` is ``[1, 2, 3, 4]``, then the first bin is ``[1, 2)`` (including
        1, but excluding 2) and the second ``[2, 3)``.  The last bin, however,
        is ``[3, 4]``, which *includes* 4.
        """
        try:
            N = len(bins)
        except TypeError:
            N = 1

        if N != 1:
            bins = [np.asarray(bins, float)]

        if range is not None:
            if len(range) == 2:
                range = [range]

        super(BinnedStatistic1D, self).__init__([x], statistic=statistic,
                                                bins=bins, range=range,
                                                mask=mask)

    @property
    def bin_edges(self):
        """
        bin_edges : 1D array of dtype float
        Return the bin edges.
        """
        return super(BinnedStatistic1D, self).bin_edges[0]

    @property
    def bin_centers(self):
        """
        bin_centers : 1D array of dtype float
        Return the bin centers.
        """
        return super(BinnedStatistic1D, self).bin_centers[0]


class BinnedStatistic2D(BinnedStatisticDD):
    """
    Compute a bidimensional binned statistic for a set of data.

    This is a generalization of a histogram2d function.  A histogram divides
    the space into bins, and returns the count of the number of points in
    each bin.  This function allows the computation of the sum, mean, median,
    or other statistic of the values within each bin.

    Parameters
    ----------
    x : (N,) array_like
        A sequence of values to be binned along the first dimension.
    y : (M,) array_like
        A sequence of values to be binned along the second dimension.
    statistic : string or callable, optional
        The statistic to compute (default is 'mean').
        The following statistics are available:

          * 'mean' : compute the mean of values for points within each bin.
            Empty bins will be represented by NaN.
          * 'median' : compute the median of values for points within each
            bin. Empty bins will be represented by NaN.
          * 'count' : compute the count of points within each bin.  This is
            identical to an unweighted histogram.  `values` array is not
            referenced.
          * 'sum' : compute the sum of values for points within each bin.
            This is identical to a weighted histogram.
          * function : a user-defined function which takes a 1D array of
            values, and outputs a single numerical statistic. This function
            will be called on the values in each bin.  Empty bins will be
            represented by function([]), or NaN if this returns an error.

    bins : int or [int, int] or array_like or [array, array], optional
        The bin specification:

          * the number of bins for the two dimensions (nx=ny=bins),
          * the number of bins in each dimension (nx, ny = bins),
          * the bin edges for the two dimensions (x_edges = y_edges = bins),
          * the bin edges in each dimension (x_edges, y_edges = bins).

    range : (2,2) array_like, optional
        The leftmost and rightmost edges of the bins along each dimension
        (if not specified explicitly in the `bins` parameters):
        [[xmin, xmax], [ymin, ymax]]. All values outside of this range will be
        considered outliers and not tallied in the histogram.
    mask : array_like
        ones and zeros with the same shape as `x`.
        Values with mask==0 will be ignored.

    See Also
    --------
    numpy.histogram2d, binned_statistic, binned_statistic_dd

    """

    def __init__(self, x, y, statistic='mean',
                 bins=10, range=None, mask=None):
        # This code is based on np.histogram2d
        try:
            N = len(bins)
        except TypeError:
            N = 1

        if N != 1 and N != 2:
            xedges = yedges = np.asarray(bins, float)
            bins = [xedges, yedges]

        super(BinnedStatistic2D, self).__init__([x, y], statistic=statistic,
                                                bins=bins, range=range,
                                                mask=mask)

    def __call__(self, values, statistic=None):
        """
        Parameters
        ----------
        values : array_like
            The values on which the statistic will be computed.  This must
            match the dimensions of ``x`` and ``y`` that were passed in when
            this object was instantiated.
        statistic : string or callable, optional
            The statistic to compute (default is whatever was passed in when
            this object was instantiated).
            The following statistics are available:

              * 'mean' : compute the mean of values for points within each bin.
                Empty bins will be represented by NaN.
              * 'median' : compute the median of values for points within each
                bin. Empty bins will be represented by NaN.
              * 'count' : compute the count of points within each bin.  This is
                identical to an unweighted histogram.  `values` array is not
                referenced.
              * 'sum' : compute the sum of values for points within each bin.
                This is identical to a weighted histogram.
              * function : a user-defined function which takes a 1D array of
                values, and outputs a single numerical statistic. This function
                will be called on the values in each bin.  Empty bins will be
                represented by function([]), or NaN if this returns an error.

        Returns
        -------
        statistic_values : array
            The values of the selected statistic in each bin.
        """
        return super(BinnedStatistic2D, self).__call__(values, statistic)


class RPhiBinnedStatistic(BinnedStatistic2D):
    """
    Create a 2-dimensional histogram by binning a 2-dimensional
    image in both radius and phi.
    """

    def __init__(self, shape, bins=10, range=None,
                 origin=None, mask=None, r_map=None, statistic='mean'):
        """
        Parameters:
        -----------
        shape : tuple of ints of length 2.
            shape of image.
        bins : int or [int, int] or array_like or [array, array], optional
            The bin specification:
            * number of bins for the two dimensions (nr=nphi=bins),
            * number of bins in each dimension (nr, nphi = bins),
            * bin edges for the two dimensions (r_edges = phi_edges = bins),
            * the bin edges in each dimension (r_edges, phi_edges = bins).
            Phi has a range of -pi to pi and is defined as arctan(row/col)
            (i.e. x is column and y is row, or "cartesian" format,
            not "matrix")
        range : (2,2) array_like, optional
            The leftmost and rightmost edges of the bins along each dimension
            (if not specified explicitly in the `bins` parameters):
            [[rmin, rmax], [phimin, phimax]]. All values outside of this range
            will be considered outliers and not tallied in the histogram.
            See "bins" parameter for definition of phi.
        origin : tuple of floats with length 2, optional
            location (in pixels) of origin (default: image center).
        mask : 2-dimensional np.ndarray of ints, optional
            array of zero/non-zero values, with shape `shape`.
            zero values will be ignored.
        r_map : 2d np.ndarray of floats, optional
            The map of pixel radii for each pixel. For example, r_map can be
            used to define the radius of each pixel relative to the origin in
            reciprocal space (on the Ewald sphere).
        statistic : string or callable, optional
            The statistic to compute (default is 'mean').
            The following statistics are available:

              * 'mean' : compute the mean of values for points within each bin.
                Empty bins will be represented by NaN.
              * 'median' : compute the median of values for points within each
                bin. Empty bins will be represented by NaN.
              * 'count' : compute the count of points within each bin.  This is
                identical to an unweighted histogram.  `values` array is not
                referenced.
              * 'sum' : compute the sum of values for points within each bin.
                This is identical to a weighted histogram.
              * function : a user-defined function which takes a 1D array of
                values, and outputs a single numerical statistic. This function
                will be called on the values in each bin.  Empty bins will be
                represented by function([]), or NaN if this returns an error.
        """
        if origin is None:
            origin = (shape[0] - 1) / 2., (shape[1] - 1) / 2.

        if r_map is None:
            r_map = radial_grid(origin, shape)

        phi_map = angle_grid(origin, shape)

        self.expected_shape = tuple(shape)
        if mask is not None:
            if mask.shape != self.expected_shape:
                raise ValueError('"mask" has incorrect shape. '
                                 ' Expected: ' + str(self.expected_shape) +
                                 ' Received: ' + str(mask.shape))
            mask = mask.reshape(-1)

        super(RPhiBinnedStatistic, self).__init__(r_map.reshape(-1),
                                                  phi_map.reshape(-1),
                                                  statistic,
                                                  bins=bins,
                                                  mask=mask,
                                                  range=range)

    def __call__(self, values, statistic=None):
        """
        Parameters
        ----------
        values : array_like
            The values on which the statistic will be computed.  This must
            match the ``shape`` that passed in when this object was
            instantiated.
        statistic : string or callable, optional
            The statistic to compute (default is whatever was passed in when
            this object was instantiated).
            The following statistics are available:

              * 'mean' : compute the mean of values for points within each bin.
                Empty bins will be represented by NaN.
              * 'median' : compute the median of values for points within each
                bin. Empty bins will be represented by NaN.
              * 'count' : compute the count of points within each bin.  This is
                identical to an unweighted histogram.  `values` array is not
                referenced.
              * 'sum' : compute the sum of values for points within each bin.
                This is identical to a weighted histogram.
              * function : a user-defined function which takes a 1D array of
                values, and outputs a single numerical statistic. This function
                will be called on the values in each bin.  Empty bins will be
                represented by function([]), or NaN if this returns an error.

        Returns
        -------
        statistic_values : array
            The values of the selected statistic in each bin.
        """
        # check for what I believe could be a common error
        if values.shape != self.expected_shape:
            raise ValueError('"values" has incorrect shape.'
                             ' Expected: ' + str(self.expected_shape) +
                             ' Received: ' + str(values.shape))
        return super(RPhiBinnedStatistic, self).__call__(values.reshape(-1),
                                                         statistic)


class RadialBinnedStatistic(BinnedStatistic1D):
    """
    Create a 1-dimensional histogram by binning a 2-dimensional
    image in radius.
    """

    def __init__(self, shape, bins=10, range=None, origin=None, mask=None,
                 r_map=None, statistic='mean'):
        """
        Parameters:
        -----------
        shape : tuple of ints of length 2.
            shape of image.
        bins : int or sequence of scalars, optional
            If `bins` is an int, it defines the number of equal-width bins in
            the given range (10 by default).  If `bins` is a sequence, it
            defines the bin edges, including the rightmost edge, allowing for
            non-uniform bin widths.  Values in `x` that are smaller than lowest
            bin edge are assigned to bin number 0, values beyond the highest
            bin are assigned to ``bins[-1]``.
            Phi has a range of -pi to pi and is defined as arctan(row/col)
            (i.e. x is column and y is row, or "cartesian" format,
            not "matrix")
        range : (float, float) or [(float, float)], optional
            The lower and upper range of the bins.  If not provided, range
            is simply ``(x.min(), x.max())``.  Values outside the range are
            ignored.
            See "bins" parameter for definition of phi.
        origin : tuple of floats with length 2, optional
            location (in pixels) of origin (default: image center).
        mask : 2-dimensional np.ndarray of ints, optional
            array of zero/non-zero values, with shape `shape`.
            zero values will be ignored.
        r_map : the map of pixel radii for each pixel. This is useful when the
            detector has some curvature or is a more complex 2D shape embedded
            in a 3D space (for example, Ewald curvature).
        statistic : string or callable, optional
            The statistic to compute (default is 'mean').
            The following statistics are available:

              * 'mean' : compute the mean of values for points within each bin.
                Empty bins will be represented by NaN.
              * 'median' : compute the median of values for points within each
                bin. Empty bins will be represented by NaN.
              * 'count' : compute the count of points within each bin.  This is
                identical to an unweighted histogram.  `values` array is not
                referenced.
              * 'sum' : compute the sum of values for points within each bin.
                This is identical to a weighted histogram.
              * function : a user-defined function which takes a 1D array of
                values, and outputs a single numerical statistic. This function
                will be called on the values in each bin.  Empty bins will be
                represented by function([]), or NaN if this returns an error.
        """
        if origin is None:
            origin = (shape[0] - 1) / 2, (shape[1] - 1) / 2

        if r_map is None:
            r_map = radial_grid(origin, shape)

        self.expected_shape = tuple(shape)
        if mask is not None:
            if mask.shape != self.expected_shape:
                raise ValueError('"mask" has incorrect shape. '
                                 ' Expected: ' + str(self.expected_shape) +
                                 ' Received: ' + str(mask.shape))
            mask = mask.reshape(-1)

        super(RadialBinnedStatistic, self).__init__(r_map.reshape(-1),
                                                    statistic,
                                                    bins=bins,
                                                    mask=mask,
                                                    range=range)

    def __call__(self, values, statistic=None):
        """
        Parameters
        ----------
        values : array_like
            The values on which the statistic will be computed.  This must
            match the ``shape`` that passed in when this object was
            instantiated.
        statistic : string or callable, optional
            The statistic to compute (default is whatever was passed in when
            this object was instantiated).
            The following statistics are available:

              * 'mean' : compute the mean of values for points within each bin.
                Empty bins will be represented by NaN.
              * 'median' : compute the median of values for points within each
                bin. Empty bins will be represented by NaN.
              * 'count' : compute the count of points within each bin.  This is
                identical to an unweighted histogram.  `values` array is not
                referenced.
              * 'sum' : compute the sum of values for points within each bin.
                This is identical to a weighted histogram.
              * function : a user-defined function which takes a 1D array of
                values, and outputs a single numerical statistic. This function
                will be called on the values in each bin.  Empty bins will be
                represented by function([]), or NaN if this returns an error.

        Returns
        -------
        statistic_values : array
            The values of the selected statistic in each bin.
        """
        # check for what I believe could be a common error
        if values.shape != self.expected_shape:
            raise ValueError('"values" has incorrect shape.'
                             ' Expected: ' + str(self.expected_shape) +
                             ' Received: ' + str(values.shape))
        return super(RadialBinnedStatistic, self).__call__(values.reshape(-1),
                                                           statistic)
