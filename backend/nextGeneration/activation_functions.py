"""Contains activation functions for nodes of the CPPN."""
import inspect
import math
import sys
import numpy as np

def get_all():
    """Returns all activation functions."""
    fns = inspect.getmembers(sys.modules[__name__])
    fns = [f[1] for f in fns if len(f)>1 and f[0] != "get_all"\
        and isinstance(f[1], type(get_all))]
    return fns

def identity(x):
    """Returns the input value."""
    return x

def sigmoid(x):
    """Returns the sigmoid of the input."""
    return 1/(1 + np.exp(-x))

def tanh(x):
    """Returns the hyperbolic tangent of the input."""
    y = np.tanh(2.5*x)
    return y



def relu(x):
    """Returns the rectified linear unit of the input."""
    return x * (x > 0)

def tanh_sig(x):
    """Returns the sigmoid of the hyperbolic tangent of the input."""
    return sigmoid(tanh(x))

def pulse(x):
    """Return the pulse fn of the input."""
    return 2*(x % 1 < .5) -1


def hat(x):
    """Returns the hat function of the input."""
    x = 1.0 - np.abs(x)
    x= np.clip(x, 0, 1)
    return x

def round_activation(x):
    """return round(x)"""
    return np.round(x) # arrays

def abs_activation(x):
    """Returns the absolute value of the input."""
    return np.abs(x)

def sqr(x):
    """Return the square of the input."""
    return np.square(x)

def elu(x):
    """Returns the exponential linear unit of the input."""
    return np.where(x > 0, x, np.exp(x) - 1)

def sin(x):
    """Returns the sine of the input."""
    y =  np.sin(x*math.pi)
    return y

def cos(x):
    """Returns the cosine of the input."""
    y =  np.cos(x*math.pi)
    return y

def gauss(x):
    """Returns the gaussian of the input."""
    y = 2*np.exp(-20.0 * (x) ** 2)-1
    return y


# Methods below are from SciPy #

#LICENSE:
# Copyright (c) 2001-2002 Enthought, Inc. 2003-2022, SciPy Developers.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


def sawtooth(t, width=1):
    """
    Taken from SciPy:
    https://github.com/scipy/scipy/blob/v1.8.0/scipy/signal/_waveforms.py#L16-L84

    Return a periodic sawtooth or triangle waveform.
    The sawtooth waveform has a period ``2*pi``, rises from -1 to 1 on the
    interval 0 to ``width*2*pi``, then drops from 1 to -1 on the interval
    ``width*2*pi`` to ``2*pi``. `width` must be in the interval [0, 1].
    Note that this is not band-limited.  It produces an infinite number
    of harmonics, which are aliased back and forth across the frequency
    spectrum.
    Parameters
    ----------
    t : array_like
        Time.
    width : array_like, optional
        Width of the rising ramp as a proportion of the total cycle.
        Default is 1, producing a rising ramp, while 0 produces a falling
        ramp.  `width` = 0.5 produces a triangle wave.
        If an array, causes wave shape to change over time, and must be the
        same length as t.
    Returns
    -------
    y : ndarray
        Output array containing the sawtooth waveform.
    Examples
    --------
    A 5 Hz waveform sampled at 500 Hz for 1 second:
    >>> t = np.linspace(0, 1, 500)
    >>> plt.plot(t, sawtooth(2 * np.pi * 5 * t))
    """
    t, w = np.asarray(t), np.asarray(width)
    w = np.asarray(w + (t - t))
    t = np.asarray(t + (w - w))
    if t.dtype.char in ['fFdD']:
        ytype = t.dtype.char # pylint: disable=unused-variable
    else:
        ytype = 'd' # pylint: disable=unused-variable
    y = np.zeros(t.shape, dtype=t.dtype)

    # width must be between 0 and 1 inclusive
    mask1 = (w > 1) | (w < 0)
    np.place(y, mask1, np.nan)

    # take t modulo 2*pi
    tmod = np.mod(t, 2 * math.pi)

    # on the interval 0 to width*2*pi function is
    #  tmod / (pi*w) - 1
    mask2 = (1 - mask1) & (tmod < w * 2 * math.pi)
    tsub = np.extract(mask2, tmod)
    wsub = np.extract(mask2, w)
    np.place(y, mask2, tsub / (math.pi * wsub) - 1)

    # on the interval width*2*pi to 2*pi function is
    #  (pi*(w+1)-tmod) / (pi*(1-w))

    mask3 = (1 - mask1) & (1 - mask2)
    tsub = np.extract(mask3, tmod)
    wsub = np.extract(mask3, w)
    np.place(y, mask3, (math.pi * (wsub + 1) - tsub) / (math.pi * (1 - wsub)))
    return y
