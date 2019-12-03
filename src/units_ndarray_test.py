import pytest
from pytest import mark

import units_ndarray as u
import numpy         as np


@mark.parametrize("units", ['kg', 'N', 's'])
def test_creates_class(units):
    u.phval(1., units)


@mark.parametrize("units", [['kg*m/s^2', 'N'], ['kg', 'eV'],['s','1/eV']])
def test_operations(units):
    x=u.phval(1, units[0])
    y=u.phval(1, units[1])
    print(x, units[0])
    print(y, units[1])
    z=x+y
    assert float(z) == float(x)+float(y)
    assert z.units == x.units == y.units
    z=x-y
    assert float(z) == float(x)-float(y)
    assert z.units == x.units == y.units
    z=x/y
    assert float(z) == float(x)/float(y)
    assert z.units == x.units - y.units
    z=x*y
    assert float(z) == float(x)*float(y)
    assert z.units == x.units + y.units


def test_plot():
    import matplotlib.pyplot as plt

    x_units='kg'
    y_units='s'

    x=u.phval(np.linspace(0.1,10,100), x_units)
    y=u.phval(np.linspace(0.1,10,100), y_units)

    plt.plot(x.val_u(x_units), y.val_u(y_units))
    plt.xlabel(x_units)
    plt.ylabel(y_units)
    #    plt.show()


