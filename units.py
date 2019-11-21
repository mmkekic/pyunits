import math as m
import os

class phval:
    """class for physical values. All physcial values are internally in 
    natural units. Every values is tuple with a float and the power of eV. All the constants 
    and units of the GNU unit program are available, to define units from strings GNU units 
    is needed"""


    def strunit(self,a,st):
        """It returs the arguments for the phval initialization with values "a" and units given 
        by the string "st" """
        cmd="units -u natural "
        out=os.popen(cmd+st).read().split()
        if out[-2]=="/":
            return (a*float(out[-3]), -1 if len(out[-1].split("^"))<2 else -int(out[-1].split("^")[-1]))
        else:
            return (a*float(out[-2]), 1 if len(out[-1].split("^"))<2 else int(out[-1].split("^")[-1]))
    
    def __init__(self, a, pw):
        """Constructor that produces a phval with value "a" and units given by "pw" if its a stirng 
        is using the function "strunit" otherwise it should be and integer and the power of eV """
        if isinstance(pw, int):
            self.v=(a,pw)
        elif isinstance(pw, str):
            self.v=self.strunit(a,pw)
        else:
            raise TypeError
        
    def __add__(self, o):
        """ Addition function, both values should have the same untis or not units, 
        with a float asumes the float to have the units of the phval"""
        if not isinstance(o,phval) and self.v[1]==0:
            return  phval(o + self.v[0], 0)
        elif isinstance(o,phval) and o.v[1]==self.v[1]:
            return  phval(o.v[0] + self.v[0], self.v[1])
        else:
            return  phval(o + self.v[0], self.v[1])
            raise TypeError

    def __radd__(self, o):
        """ right addition function, both values should have the same untis,
        with a float asumes the float to have the units of the phval"""
        if not isinstance(o,phval) and self.v[1]==0:
            return  phval(o + self.v[0], 0)
        elif isinstance(o,phval) and o.v[1]==self.v[1]:
            return  phval(o.v[0] + self.v[0], self.v[1])
        else:
            return  phval(o + self.v[0], self.v[1])
            raise TypeError

    def __sub__(self, o):
        """ substraction, both values should have the same untis or not units,
        with a float asumes the float to have the units of the phval"""
        if not isinstance(o,phval) and self.v[1]==0:
            return  phval(self.v[0] - o, 0)
        elif isinstance(o,phval) and o.v[1]==self.v[1]:
            return  phval(self.v[0] - o.v[0], self.v[1])
        else:
            return  phval(self.v[0] - o, self.v[1])
            raise TypeError

    def __rsub__(self, o):
        """ right addition function, both values should have the same untis, 
        with a float asumes the float to have the units of the phval"""
        if not isinstance(o,phval) and self.v[1]==0:
            return  phval(o - self.v[0], 0)
        elif isinstance(o,phval) and o.v[1]==self.v[1]:
            return  phval(o.v[0] - self.v[0], self.v[1])
        else:
            return  phval(o - self.v[0], self.v[1])
            raise TypeError
            
    def __mul__(self, o):
        """ multiplication between phvals or a float and phval """
        if isinstance(o, phval):
            return phval(self.v[0]*o.v[0], self.v[1]+o.v[1])
        else:
            return  phval(self.v[0]*o, self.v[1])
    __rmul__=__mul__
        
    def __truediv__(self, o):
        """ division between phvals or a float and phval """
        if isinstance(o, phval):
            return  phval(self.v[0]/o.v[0], self.v[1]-o.v[1])
        else:
            return  phval(self.v[0]/o, self.v[1])

    def __rtruediv__(self, o):
        """ right division between phvals or a float and phval """
        if isinstance(o, phval):
            return  phval(o.v[0]/self.v[0], o.v[1] - self.v[1])
        else:
            return  phval(o/self.v[0], -self.v[1])

    def __neg__(self):
        return phval(-self.v[0], self.v[1])

    def __pos__(self):
        return self

    def __invert__(self):
        return phval(1.0/self.v[0], -self.v[1])
    def sqrt(self):
        if self.v[1]%2==0:
            return phval(m.sqrt(self.v[0]), int(self.v[1]/2))


        
    def __pow__(self, pw):
        """ power funtion, only  integer powers are valid for pval with units """
        if isinstance(pw, int):
            return  phval(self.v[0]**pw, self.v[1]*pw)
        elif self.v[1]==0:
            return  phval(self.v[0]**pw, 0)
        else:
            raise TypeError
    def __lt__(self, o):
        if isinstance(o,phval) and self.v[1]==o.v[1]:
            return (self.v[0]<o.v[0])
        elif not isinstance(o,phval):
            return (self.v[0]<o)
        else:
            raise TypeError

    def __le__(self, o):
        if isinstance(o,phval) and self.v[1]==o.v[1]:
            return (self.v[0]<=o.v[0])
        elif not isinstance(o,phval):
            return (self.v[0]<=o)
        else:
            raise TypeError
    
    def __gt__(self, o):
        if isinstance(o,phval) and self.v[1]==o.v[1]:
            return (self.v[0]>o.v[0])
        elif not isinstance(o,phval):
            return (self.v[0]>o)
        else:
            raise TypeError

    def __ge__(self, o):
        if isinstance(o,phval) and self.v[1]==o.v[1]:
            return (self.v[0]>=o.v[0])
        elif not isinstance(o,phval):
            return (self.v[0]>=o)
        else:
            raise TypeError

    def __eq__(self, o):
        if isinstance(o,phval) and self.v[1]==o.v[1]:
            return (self.v[0]==o.v[0])
        elif not isinstance(o,phval):
            return (self.v[0]==o)
        else:
            raise TypeError

    def __ne__(self, o):
        if isinstance(o,phval) and self.v[1]==o.v[1]:
            return (self.v[0]!=o.v[0])
        elif not isinstance(o,phval):
            return (self.v[0]!=o)
        else:
            raise TypeError
    
    def __float__(self):
        return float(self.v[0])

    def __float64__(self):
        return float(self.v[0])

    def __int__(self):
        return int(self.v[0])

    def __isub__(self, o):
        if not isinstance(o,phval):
            self.v[0] -= o
        elif o.v[1]==self.v[1]:
            self.v[0] -= o.v[0]



    
    def sin(self):
        if self.v[1]==0:
            return phval(m.sin(self.v[0]),0)
        else:
            raise TypeError
    def cos(self):
        if self.v[1]==0:
            return phval(m.cos(self.v[0]),0)
        else:
            raise TypeError
    def tan(self):
        if self.v[1]==0:
            return phval(m.tan(self.v[0]),0)
        else:
            raise TypeError
    def exp(self):
        if self.v[1]==0:
            return phval(m.exp(self.v[0]),0)
        else:
            raise TypeError
    def log(self):
        if self.v[1]==0:
            return phval(m.log(self.v[0]),0)
        else:
            raise TypeError
    def log10(self):
        if self.v[1]==0:
            return phval(m.log10(self.v[0]),0)
        else:
            raise TypeError
        
    def arcsin(self):
        if self.v[1]==0:
            return phval(m.asin(self.v[0]),0)
        else:
            raise TypeError
    def arctan(self):
        if self.v[1]==0:
            return phval(m.atan(self.v[0]),0)
        else:
            raise TypeError

    def arccos(self):
        if self.v[1]==0:
            return phval(m.acos(self.v[0]),0)
        else:
            raise TypeError

        
    def fl(self,name=None):
        """Returns the float value, without the units"""
        if isinstance(name, str):
            un=phval(1,unit)         
            return self.v[0]/un.v[0]
        elif name==None:
            return self.v[0]
        else:
            raise TypeError

        return self.v[0]

    def units(self):
        """Returns the units, allways a power of eV"""
        return "eV^"+str(self.v[1])

    def __str__(self):
        """Default string function allways in eV"""
        return str(self.v[0]) if self.v[1]==0 else str(self.v[0]) + " eV^"+str(self.v[1])

    def str(self, unit):
        """returns the value with the units labeled by the string "unit" """
        if isinstance(unit, str):
            un=phval(1,unit)         
            if self.v[1]==un.v[1]:
                return str(self.v[0]/un.v[0]) +" "+ unit
            else:
                raise TypeError
        else:
            raise TypeError
    
        
    def unit(self):
        """Returns the phval with value one for the given unit"""
        return phval(1.0, self.v[1])
    
    def unitless(f, *args, output_units=None):
        """ Function that returns the a funciton without units, this can be usefull to 
        contruct standard functions to be used by other libraries. The arguments are: the 
        function, the units for the function, and the otput_units it a value with other 
        that eV^n are needed.""" 
        fctor=phval(1.0,0)
        if output_units!=None:
            fctor=f(*args)
            if output_units.v[1] != fctor.v[1]:
                raise Exception
        
        def fout(*nou_args):
            out = f(*tuple(l * r for l, r in zip(args,nou_args)))
            return out.fl()/fctor.v[0]
        return fout


