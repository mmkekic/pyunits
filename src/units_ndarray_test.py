import pytest
from pytest import mark
import units_ndarray as u

@mark.parametrize("units", ['kg', 'N', 's'])
def test_creates_class(units):
    u.phval(1., units)
