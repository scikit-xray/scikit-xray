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
from __future__ import absolute_import, division, print_function
import logging

import numpy as np
from numpy.testing import assert_equal, assert_array_almost_equal
import pytest

import skbeam.core.utils as utils
from skbeam.core.correlation import (
    multi_tau_auto_corr,
    auto_corr_scat_factor,
    lazy_one_time,
    lazy_two_time,
    two_time_corr,
    two_time_state_to_results,
    one_time_from_two_time,
    CrossCorrelator,
    get_four_time_from_two_time,
)
from skbeam.core.mask import bad_to_nan_gen
from skbeam.core.roi import ring_edges, segmented_rings


logger = logging.getLogger(__name__)


def setup():
    global num_levels, num_bufs, xdim, ydim, stack_size, img_stack, rois
    num_levels = 6
    num_bufs = 4  # must be even
    xdim = 256
    ydim = 512
    stack_size = 100
    img_stack = np.random.randint(1, 3, (stack_size, xdim, ydim))
    rois = np.zeros_like(img_stack[0])
    # make sure that the ROIs can be any integers greater than 1.
    # They do not have to start at 1 and be continuous
    rois[0: xdim // 10, 0: ydim // 10] = 5
    rois[xdim // 10: xdim // 5, ydim // 10: ydim // 5] = 3


def test_lazy_vs_original():
    setup()
    # run the correlation on the full stack
    full_gen_one = lazy_one_time(img_stack, num_levels, num_bufs, rois)
    for gen_state_one in full_gen_one:
        pass

    g2, lag_steps = multi_tau_auto_corr(num_levels, num_bufs, rois, img_stack)

    assert np.all(g2 == gen_state_one.g2)
    assert np.all(lag_steps == gen_state_one.lag_steps)

    full_gen_two = lazy_two_time(rois, img_stack, stack_size, num_bufs, num_levels)
    for gen_state_two in full_gen_two:
        pass
    final_gen_result_two = two_time_state_to_results(gen_state_two)

    two_time = two_time_corr(rois, img_stack, stack_size, num_bufs, num_levels)

    assert np.all(two_time[0] == final_gen_result_two.g2)
    assert np.all(two_time[1] == final_gen_result_two.lag_steps)


def test_lazy_two_time():
    setup()
    # run the correlation on the full stack
    full_gen = lazy_two_time(rois, img_stack, stack_size, stack_size, 1)
    for full_state in full_gen:
        pass
    final_result = two_time_state_to_results(full_state)

    # make sure we have essentially zero correlation in the images,
    # since they are random integers
    assert np.average(final_result.g2 - 1) < 0.01

    # run the correlation on the first half
    gen_first_half = lazy_two_time(
        rois,
        img_stack[: stack_size // 2],
        stack_size,
        num_bufs=stack_size,
        num_levels=1,
    )
    for first_half_state in gen_first_half:
        pass
    # run the correlation on the second half by passing in the state from the
    # first half
    gen_second_half = lazy_two_time(
        rois,
        img_stack[stack_size // 2:],
        stack_size,
        num_bufs=stack_size,
        num_levels=1,
        two_time_internal_state=first_half_state,
    )

    for second_half_state in gen_second_half:
        pass
    result = two_time_state_to_results(second_half_state)

    assert np.all(full_state.g2 == result.g2)


def test_lazy_one_time():
    setup()
    # run the correlation on the full stack
    full_gen = lazy_one_time(img_stack, num_levels, num_bufs, rois)
    for full_result in full_gen:
        pass

    # make sure we have essentially zero correlation in the images,
    # since they are random integers
    assert np.average(full_result.g2 - 1) < 0.01

    # run the correlation on the first half
    gen_first_half = lazy_one_time(
        img_stack[: stack_size // 2], num_levels, num_bufs, rois
    )
    for first_half_result in gen_first_half:
        pass
    # run the correlation on the second half by passing in the state from the
    # first half
    gen_second_half = lazy_one_time(
        img_stack[stack_size // 2:],
        num_levels,
        num_bufs,
        rois,
        internal_state=first_half_result.internal_state,
    )

    for second_half_result in gen_second_half:
        pass

    assert np.all(full_result.g2 == second_half_result.g2)


def test_two_time_corr():
    setup()
    y = []
    for i in range(50):
        y.append(img_stack[0])
    two_time = two_time_corr(rois, np.asarray(y), 50, num_bufs=50, num_levels=1)
    assert np.all(two_time[0])

    # check the number of buffers are even
    with pytest.raises(ValueError):
        two_time_corr(rois, np.asarray(y), 50, num_bufs=25, num_levels=1)


def test_auto_corr_scat_factor():
    num_levels, num_bufs = 3, 4
    tot_channels, lags, dict_lags = utils.multi_tau_lags(num_levels, num_bufs)
    beta = 0.5
    relaxation_rate = 10.0
    baseline = 1.0

    g2 = auto_corr_scat_factor(lags, beta, relaxation_rate, baseline)

    assert_array_almost_equal(
        g2, np.array([1.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), decimal=8
    )


def test_bad_images():
    setup()
    g2, lag_steps = multi_tau_auto_corr(4, num_bufs, rois, img_stack)
    # introduce bad images
    bad_img_list = [3, 21, 35, 48]
    # convert each bad image to np.nan array
    images = bad_to_nan_gen(img_stack, bad_img_list)

    # then use new images (including bad images)
    g2_n, lag_steps_n = multi_tau_auto_corr(4, num_bufs, rois, images)

    assert_array_almost_equal(g2[:, 0], g2_n[:, 0], decimal=3)
    assert_array_almost_equal(g2[:, 1], g2_n[:, 1], decimal=3)


def test_one_time_from_two_time():
    np.random.seed(333)
    num_lev = 1
    num_buf = 10  # must be even
    x_dim = 10
    y_dim = 10
    stack = 10
    imgs = np.random.randint(1, 3, (stack, x_dim, y_dim))
    imgs[:, 0, 0] += 5
    roi = np.zeros_like(imgs[0])
    # make sure that the ROIs can be any integers greater than 1.
    # They do not have to start at 1 and be continuous
    roi[0: x_dim // 5, 0: y_dim // 10] = 5
    roi[x_dim // 10: x_dim // 5 + 1, y_dim // 10: y_dim // 5] = 3

    g2, lag_steps, _state = two_time_corr(roi, imgs, stack, num_buf, num_lev)
    g2_t, _ = multi_tau_auto_corr(num_lev, num_buf, roi, imgs)

    one_time, error_one_time = one_time_from_two_time(g2, calc_errors=True)
    assert_array_almost_equal(
        one_time,
        np.array(
            [
                [1.02222222, 1.0, 1.0, 1.0, 0.98148148, 1.0, 1.0, 1.0, 1.0, 1.0],
                [
                    1.37103962,
                    1.3595679,
                    1.35260377,
                    1.34863946,
                    1.36706349,
                    1.36235828,
                    1.35813492,
                    1.37840136,
                    1.36607143,
                    1.35714286,
                ],
            ]
        ),
    )
    assert_array_almost_equal(
        one_time[0].mean() / one_time[1].mean(),
        g2_t.T[0].mean() / g2_t.T[1].mean(),
        decimal=2,
    )

    assert_array_almost_equal(
        error_one_time[0][1:],
        np.array(
            [
                np.std(np.diagonal(g2[0], offset=x)) / np.sqrt(x)
                for x in range(1, g2[0].shape[0])
            ]
        ),
        decimal=2,
    )


@pytest.mark.skipif(
    int(np.__version__.split(".")[1]) > 14, reason="Test is numerically unstable"
)
def test_CrossCorrelator1d():
    """ Test the 1d version of the cross correlator with these methods:
        -method='regular', no mask
        -method='regular', masked
        -method='symavg', no mask
        -method='symavg', masked
     """
    np.random.seed(123)
    # test 1D data
    sigma = 0.1
    Npoints = 100
    x = np.linspace(-10, 10, Npoints)

    sigma = 0.2
    # purposely have sparsely filled values (with lots of zeros)
    peak_positions = (np.random.random(10) - 0.5) * 20
    y = np.zeros_like(x)
    for peak_position in peak_positions:
        y += np.exp(-((x - peak_position) ** 2) / 2.0 / sigma ** 2)

    mask_1D = np.ones_like(y)
    mask_1D[10:20] = 0
    mask_1D[60:90] = 0
    mask_1D[111:137] = 0
    mask_1D[211:237] = 0
    mask_1D[411:537] = 0

    mask_1D *= mask_1D[::-1]

    cc1D = CrossCorrelator(mask_1D.shape)
    cc1D_symavg = CrossCorrelator(mask_1D.shape, normalization="symavg")
    cc1D_masked = CrossCorrelator(mask_1D.shape, mask=mask_1D)
    cc1D_masked_symavg = CrossCorrelator(
        mask_1D.shape, mask=mask_1D, normalization="symavg"
    )

    assert_equal(cc1D.nids, 1)

    ycorr_1D = cc1D(y)
    ycorr_1D_masked = cc1D_masked(y * mask_1D)
    ycorr_1D_symavg = cc1D_symavg(y)
    ycorr_1D_masked_symavg = cc1D_masked_symavg(y * mask_1D)

    assert_array_almost_equal(
        ycorr_1D[::20],
        np.array(
            [
                -1.155123e-14,
                6.750373e-03,
                6.221636e-01,
                7.105527e-01,
                1.187275e00,
                2.984563e00,
                1.092725e00,
                1.198341e00,
                1.045922e-01,
                5.451511e-06,
            ]
        ),
    )
    assert_array_almost_equal(
        ycorr_1D_masked[::20],
        np.array(
            [
                -5.172377e-16,
                np.nan,
                7.481473e-01,
                6.066887e-02,
                4.470989e-04,
                2.330335e00,
                np.nan,
                7.109758e-01,
                np.nan,
                2.275846e-14,
            ]
        ),
    )

    assert_array_almost_equal(
        ycorr_1D_symavg[::20],
        np.array(
            [
                -5.3002753,
                1.54268227,
                0.86220476,
                0.57715207,
                0.86503802,
                2.94383202,
                0.7587901,
                0.99763715,
                0.16800951,
                1.23506293,
            ]
        ),
    )

    assert_array_almost_equal(
        ycorr_1D_masked_symavg[::20][:-1],
        np.array(
            [
                -5.30027530e-01,
                np.nan,
                1.99940257e00,
                7.33127871e-02,
                1.00000000e00,
                2.15887870e00,
                np.nan,
                9.12832602e-01,
                np.nan,
            ]
        ),
    )


def test_CrossCorrelator2d():
    """ Test the 2D case of the cross correlator.
        With non-binary labels.
    """
    np.random.seed(123)
    # test 2D data
    Npoints2 = 10
    x2 = np.linspace(-10, 10, Npoints2)
    X, Y = np.meshgrid(x2, x2)
    Z = np.random.random((Npoints2, Npoints2))

    np.random.seed(123)
    sigma = 0.2
    # purposely have sparsely filled values (with lots of zeros)
    # place peaks in random positions
    peak_positions = (np.random.random((2, 10)) - 0.5) * 20
    for peak_position in peak_positions:
        Z += np.exp(
            -((X - peak_position[0]) ** 2 + (Y - peak_position[1]) ** 2)
            / 2.0
            / sigma ** 2
        )

    mask_2D = np.ones_like(Z)
    mask_2D[1:2, 1:2] = 0
    mask_2D[7:9, 4:6] = 0
    mask_2D[1:2, 9:] = 0

    # Compute with segmented rings
    edges = ring_edges(1, 3, num_rings=2)
    segments = 5
    x0, y0 = np.array(mask_2D.shape) // 2

    maskids = segmented_rings(edges, segments, (y0, x0), mask_2D.shape)

    cc2D_ids = CrossCorrelator(mask_2D.shape, mask=maskids)
    cc2D_ids_symavg = CrossCorrelator(
        mask_2D.shape, mask=maskids, normalization="symavg"
    )

    # 10 ids
    assert_equal(cc2D_ids.nids, 10)

    ycorr_ids_2D = cc2D_ids(Z)
    ycorr_ids_2D_symavg = cc2D_ids_symavg(Z)
    index = 0
    ycorr_ids_2D[index][ycorr_ids_2D[index].shape[0] // 2]
    assert_array_almost_equal(
        ycorr_ids_2D[index][ycorr_ids_2D[index].shape[0] // 2],
        np.array([1.22195059, 1.08685771, 1.43246508, 1.08685771, 1.22195059]),
    )

    index = 1
    ycorr_ids_2D[index][ycorr_ids_2D[index].shape[0] // 2]
    assert_array_almost_equal(
        ycorr_ids_2D[index][ycorr_ids_2D[index].shape[0] // 2],
        np.array([1.24324268, 0.80748997, 1.35790022, 0.80748997, 1.24324268]),
    )

    index = 0
    ycorr_ids_2D_symavg[index][ycorr_ids_2D[index].shape[0] // 2]
    assert_array_almost_equal(
        ycorr_ids_2D_symavg[index][ycorr_ids_2D[index].shape[0] // 2],
        np.array([0.84532695, 1.16405848, 1.43246508, 1.16405848, 0.84532695]),
    )

    index = 1
    ycorr_ids_2D_symavg[index][ycorr_ids_2D[index].shape[0] // 2]
    assert_array_almost_equal(
        ycorr_ids_2D_symavg[index][ycorr_ids_2D[index].shape[0] // 2],
        np.array([0.94823482, 0.8629459, 1.35790022, 0.8629459, 0.94823482]),
    )


def test_get_four_time_from_two_time():
    np.random.seed(0)
    ll = 5
    g12 = np.random.rand(2, ll, ll)
    for k in range(g12.shape[0]):
        q = g12[k]
        g12[k] = np.tril(q) + np.tril(q).T - 2 * np.diag(np.diag(q)) + \
            np.diag(np.ones(len(q)))
    g2 = np.random.rand(2, ll) + 10 * np.exp(-0.05 * np.array(range(ll)))
    res = get_four_time_from_two_time(g12, g2, (1, ll))

    assert_array_almost_equal(res, [[0., 0.00019202, 0.00053148, 0.],
                                    [0., 0.00031959, 0.00028276, 0.]],
                              decimal=8)


def test_CrossCorrelator_badinputs():
    with pytest.raises(ValueError):
        CrossCorrelator((1, 1, 1))

    with pytest.raises(ValueError):
        cc = CrossCorrelator((10, 10))
        a = np.ones((10, 11))
        cc(a)

    with pytest.raises(ValueError):
        cc = CrossCorrelator((10, 10))
        a = np.ones((10, 10))
        a2 = np.ones((10, 11))
        cc(a, a2)
