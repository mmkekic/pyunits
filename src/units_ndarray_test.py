import pytest
from pytest import mark

import units_ndarray as u
import numpy         as np


@mark.parametrize("units", ['kg', 'N', 's'])
def test_creates_class(units):
    u.phval(1., units)


@mark.parametrize("units", [['kg*m/s^2', 'N']])
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

print(u.phval.strunit(1.,"N"))
