'''
For isotropic material,  its elastic response can be completely described
by two constant scalar material properties, e.g., the Young's
modulus Es and Poisson's ratio vs.
'''
import numpy as np
import matplotlib.pyplot as plt

class MicroStruct:
    def __init__(self, Es, vs) -> None:
        self.Es = Es
        self.vs = vs
        self.Gs = Es / (2 * (1+vs))
    def get_moduli(self):
        pass
   


class IsoTruss(MicroStruct):
    def __init__(self, Es, vs):
        super().__init__(Es, vs)

    def get_moduli(self, rho):
        deriv = 0
        Es = self.Es
        vs = self.vs
        Gs = self.Gs

        E = Es * (( 2.05292e-01 - 3.30265e-02*vs) * (rho**(1-deriv)) * (1+0*deriv) + \
	     ( 8.12145e-02 + 2.72431e-01*vs) * (rho**(2-deriv)) * (1+1*deriv) + \
	     ( 6.49737e-01 - 2.42374e-01*vs) * (rho**(3-deriv)) * (1+2*deriv))

        v = ( 2.47760e-01 + 1.69804e-02*vs) * (1-deriv) + \
	     (-1.59293e-01 + 7.38598e-01*vs) * (rho**(1-deriv)) * (1+0*deriv) + \
	     (-1.86279e-01 - 4.83229e-01*vs) * (rho**(2-deriv)) * (1+1*deriv) + \
	     ( 9.77457e-02 + 7.26595e-01*vs) * (rho**(3-deriv)) * (1+2*deriv)

        G = Gs * (( 1.63200e-01 + 1.27910e-01*vs) * (rho**(1-deriv)) * (1+0*deriv) + \
	     ( 6.00810e-03 + 4.13331e-01*vs) * (rho**(2-deriv)) * (1+1*deriv) + \
	     ( 7.22847e-01 - 3.56032e-01*vs) * (rho**(3-deriv)) * (1+2*deriv))
    
        return (E, v, G)

    def get_CE(self):pass

class OctetTruss:
    pass

class ORCTruss:
    pass



if __name__ == "__main__":
    m1 = IsoTruss(Es=1,vs=0.2)
    m2 = IsoTruss(Es=1,vs=0.3)
    m3 = IsoTruss(Es=1,vs=0.4)
    x = np.linspace(0, 1, 101)
    y1 = m1.get_moduli(x)
    y2 = m2.get_moduli(x)
    y3 = m3.get_moduli(x)
    fig, ax = plt.subplots()
    ax.plot(x, y1[2], linewidth=2.0)       
    ax.plot(x, y2[2], '--', linewidth=2.0)   
    ax.plot(x, y3[2], '-.',linewidth=2.0)   

    plt.show()