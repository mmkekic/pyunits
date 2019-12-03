import numpy as np
import os

COMPARISON_UFUNC = {np.greater, np.greater_equal, np.less, np.less_equal, np.not_equal, np.equal}
UNITLESS_UFUNC   = {np.sin, np.cos, np.exp, np.arcsin, np.arccos, np.tan, np.arctan, np.log, np.log10}

class UnitError(Exception) : pass

class phval(np.ndarray):
    def __new__(cls, input_array, units=None):
        if isinstance(units, int):
            units = units
            obj   = input_array
        elif isinstance(units, str):
            val, pw = cls.strunit(1., units)
            obj     = np.multiply(input_array, val)
            units   = pw
        else:
            raise TypeError

        obj = np.asarray(obj).view(cls)
        obj.units = units
        return obj

    @property
    def values(self):
        return self.view(np.ndarray)

    def val_u(self, uname):
        if uname:
            return (self/phval(1., uname)).view(np.ndarray)

    @staticmethod
    def strunit(a,st):
        """It returs the arguments for the phval initialization with values "a" and units given 
        by the string "st" """
        cmd="units -u natural "
        out=os.popen(cmd+st).read().split()
        if out[-2]=="/":
            return (a*float(out[-3]), -1 if len(out[-1].split("^"))<2 else -int(out[-1].split("^")[-1]))
        else:
            return (a*float(out[-2]), 1 if len(out[-1].split("^"))<2 else int(out[-1].split("^")[-1]))

    def __repr__(self):
        return  (self.values.__repr__()) + " eV^"+str(self.units)

    def __str__(self):
        return  (self.values.__str__()) + " eV^"+str(self.units)

    def __getitem__(self, indx):
        return phval(self.values[indx], self.units)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        list_inputs = list(inputs)
        list_units = [inp.units if isinstance(inp, phval) else 0 for inp in inputs]
        casted_inputs = [inp.view(np.ndarray) if isinstance(inp, phval) else inp for inp in inputs]


        if ufunc == np.power:
            units = list_units[0] * inputs[1]
            if type(units)==int:
                return phval(ufunc(*casted_inputs), units=units)
            else:
                raise UnitError("Values with units only have integer powers")

        if ufunc in UNITLESS_UFUNC:
            units = list_units[0]
            if units==0:
                return phval(ufunc(*casted_inputs), units=units)
            else:
                raise UnitError("This is a unitless fucntion, divide by some physical scale!")


        if ufunc == np.negative:
            return phval(ufunc(*casted_inputs), units=list_units[0])

        if ufunc in COMPARISON_UFUNC.union([np.add, np.subtract]):
            if (list_units[0] == list_units[1]) or any(np.all(inputs == 0) for inputs in casted_inputs):
                return phval(ufunc(*casted_inputs), units=list_units[0])
            else:
                raise UnitError("Can't compare non-zero values with different units!")
        if ufunc == np.multiply:
            units = list_units[0] + list_units[1]
            return phval(ufunc(*casted_inputs), units=units)

        if ufunc == np.divide:
            units = list_units[0] - list_units[1]
            return phval(ufunc(*casted_inputs), units=units)

        else:
            raise TypeError

    def __array_finalize__(self, obj):
        if obj is None: return
        self.units = getattr(obj, 'units', None)
