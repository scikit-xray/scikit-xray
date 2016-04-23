from skbeam.core.accumulators.projector import RadialProjector
from numpy.testing import assert_array_almost_equal
import numpy as np


class TestRadialProjector(object):

    def setup(self):

        # Create test image - a sinc function.
        # Integrating in phi will produce sin(x)

        xsize, ysize = (900, 1024)

        xarr = np.arange(xsize)
        yarr = np.arange(ysize)
        xgrid, ygrid = np.meshgrid(yarr, xarr)
        self.rgrid = np.sqrt(xgrid**2 + ygrid**2)
        # use 100 here to make the function go through many periods
        self.image = np.sinc(self.rgrid / 100.)

    def testRadialProjector(self):

        params = [[self.image.shape[0], self.image.shape[1], False],
                  [self.image.shape[1], self.image.shape[0], True]]
        for xsize, ysize, cartesian in params:
            radproj = RadialProjector(xsize, ysize, xc=0, yc=0, rmin=100,
                                      rmax=900, nbins=100, phimin=5, phimax=60,
                                      norm=False, cartesian=cartesian)
            projection = radproj(self.image)
            projection /= projection.max()
            ref = np.sin(radproj.bin_centers/100.*np.pi)
            # the projection won't be precisely equal to the analytic
            # np.sin formula because the pixel r-values are quantized.
            assert_array_almost_equal(ref, projection, decimal=2)
