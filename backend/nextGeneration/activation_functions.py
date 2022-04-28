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


def sawtooth(t, width=1):
    """
    Taken from scipy
    From https://github.com/scipy/scipy/blob/v1.8.0/scipy/signal/_waveforms.py#L16-L84
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
        ytype = t.dtype.char
    else:
        ytype = 'd'
    y = np.zeros(t.shape, ytype)

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

def tanh_sig(x):
    """Returns the sigmoid of the hyperbolic tangent of the input."""
    return sigmoid(tanh(x))

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
