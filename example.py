import units as u
eV = u.phval(1,"eV")
kg = u.phval(1,"kg")
Msun = u.phval(1,"solarmass")
cm = u.phval(1,"cm")
m = u.phval(1,"m")
GeV = u.phval(1,"GeV")
kpc=u.phval(1,"kpc")

#function that gives the density of DM
rhoH = 1.4e7*Msun/(kpc)**3
RH=16.1*kpc
def rho_NFW(r):
    return rhoH/((1.+r/RH)**2*r/RH)

#Denity at earth approx 8kpc

print("\nDensity at earth: ",rho_NFW(8*kpc))

#or in a more understandable units:

print("With more standard units: ", rho_NFW(8*kpc).str("GeV/cm^3"))

