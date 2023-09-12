'''
For isotropic material,  its elastic response can be completely described
by two constant scalar material properties, e.g., the Young's
modulus Es and Poisson's ratio vs.
'''
import numpy as np
from Material import Material
class MicroStruct(Material):
    def __init__(self, Es, vs) -> None:
        super().__init__(Es, vs)
    def get_CE(self):
        raise NotImplementedError('get_CE must be defined')
    def get_CE_deriv(self):
        raise NotImplementedError('get_CE_deriv must be defined')


class SIMP(MicroStruct):
    def __init__(self, Es, vs):
        super().__init__(Es, vs)

    def _get_moduli(self, p):
        deriv = 0
        Es = self.Es
        vs = self.vs
        Gs = self.Gs

        E = Es * p ** (3-deriv) * (1+2*deriv)
        v = vs * (1-deriv)
        G = Gs * p ** (3-deriv) * (1+2*deriv)
        return (E, v, G)

    def _get_moduli_deriv(self, p):
        deriv = 1
        Es = self.Es
        vs = self.vs
        Gs = self.Gs

        DE = Es * p ** (3-deriv) * (1+2*deriv)
        Dv = vs * (1-deriv)
        DG = Gs * p ** (3-deriv) * (1+2*deriv)
        return (DE, Dv, DG)  

    def get_CE(self, p):
        E,v,G = self._get_moduli(p)
        C1111 = E * (1.0 - v) / (1.0 - v - 2*v**2)
        C1122 = (E * v) / (1.0 - v - 2*v**2)
        C1212 = G
        CE = [C1111,   0,     0,     0,   C1122,   0,     0,     0,   C1122, \
            0,   C1212,   0,   C1212,   0,     0,     0,     0,     0,  \
            0,     0,   C1212,   0,     0,     0,   C1212,   0,     0,  \
            0,   C1212,   0,   C1212,   0,     0,     0,     0,     0,  \
        C1122,   0,     0,     0,   C1111,   0,     0,     0,   C1122,\
            0,     0,     0,    0,     0,   C1212,   0,   C1212,   0,  \
            0,     0,   C1212,   0,     0,     0,   C1212,   0,     0,  \
            0,     0,     0 ,    0,     0,   C1212,   0,   C1212,   0,  \
        C1122,   0,     0,     0,   C1122,  0,     0,     0,   C1111]
        CE = np.array(CE).reshape((9,9))
        return CE

    def get_CE_deriv(self, p):
        E,v,G = self._get_moduli(p)
        DE,Dv,DG = self._get_moduli_deriv(p)
        C1111 = ((DE*(1-v)-E*Dv)*(1-v-2*v**2)-E*(1-v)*(-Dv-4*v*Dv)) / (1-v-2*v**2)**2
        C1122 = ((DE*v+E*Dv)*(1-v-2*v**2)-E*v*(-Dv-4*v*Dv)) / (1-v-2*v**2)**2
        C1212 = DG
        CE = [C1111,   0,     0,     0,   C1122,   0,     0,     0,   C1122, \
            0,   C1212,   0,   C1212,   0,     0,     0,     0,     0,  \
            0,     0,   C1212,   0,     0,     0,   C1212,   0,     0,  \
            0,   C1212,   0,   C1212,   0,     0,     0,     0,     0,  \
        C1122,   0,     0,     0,   C1111,   0,     0,     0,   C1122,\
            0,     0,     0,    0,     0,   C1212,   0,   C1212,   0,  \
            0,     0,   C1212,   0,     0,     0,   C1212,   0,     0,  \
            0,     0,     0 ,    0,     0,   C1212,   0,   C1212,   0,  \
        C1122,   0,     0,     0,   C1122,  0,     0,     0,   C1111]
        CE_deriv = np.array(CE).reshape((9,9))
        return CE_deriv


class IsoTruss(SIMP):
    def __init__(self, Es, vs):
        super().__init__(Es, vs)

    def _get_moduli(self, p):
        deriv = 0
        Es = self.Es
        vs = self.vs
        Gs = self.Gs

        E = Es * (( 2.05292e-01 - 3.30265e-02*vs) * (p**(1-deriv)) * (1+0*deriv) + 
	     ( 8.12145e-02 + 2.72431e-01*vs) * (p**(2-deriv)) * (1+1*deriv) +
	     ( 6.49737e-01 - 2.42374e-01*vs) * (p**(3-deriv)) * (1+2*deriv))

        v =( 2.47760e-01 + 1.69804e-02*vs) * (1-deriv) + \
	     (-1.59293e-01 + 7.38598e-01*vs) * (p**(1-deriv)) * (1+0*deriv) + \
	     (-1.86279e-01 - 4.83229e-01*vs) * (p**(2-deriv)) * (1+1*deriv) + \
	     ( 9.77457e-02 + 7.26595e-01*vs) * (p**(3-deriv)) * (1+2*deriv)

        G = Gs * (( 1.63200e-01 + 1.27910e-01*vs) * (p**(1-deriv)) * (1+0*deriv) + 
	     ( 6.00810e-03 + 4.13331e-01*vs) * (p**(2-deriv)) * (1+1*deriv) + 
	     ( 7.22847e-01 - 3.56032e-01*vs) * (p**(3-deriv)) * (1+2*deriv))
    
        return (E, v, G)

    def _get_moduli_deriv(self, p):
        deriv = 1
        Es = self.Es
        vs = self.vs
        Gs = self.Gs

        DE = Es * (( 2.05292e-01 - 3.30265e-02*vs) * (p**(1-deriv)) * (1+0*deriv) + 
	     ( 8.12145e-02 + 2.72431e-01*vs) * (p**(2-deriv)) * (1+1*deriv) +
	     ( 6.49737e-01 - 2.42374e-01*vs) * (p**(3-deriv)) * (1+2*deriv))

        Dv =( 2.47760e-01 + 1.69804e-02*vs) * (1-deriv) + \
	     (-1.59293e-01 + 7.38598e-01*vs) * (p**(1-deriv)) * (1+0*deriv) + \
	     (-1.86279e-01 - 4.83229e-01*vs) * (p**(2-deriv)) * (1+1*deriv) + \
	     ( 9.77457e-02 + 7.26595e-01*vs) * (p**(3-deriv)) * (1+2*deriv)

        DG = Gs * (( 1.63200e-01 + 1.27910e-01*vs) * (p**(1-deriv)) * (1+0*deriv) + 
	     ( 6.00810e-03 + 4.13331e-01*vs) * (p**(2-deriv)) * (1+1*deriv) + 
	     ( 7.22847e-01 - 3.56032e-01*vs) * (p**(3-deriv)) * (1+2*deriv))
        return (DE, Dv, DG)  

class OctetTruss(SIMP):
    def __init__(self, Es, vs):
        super().__init__(Es, vs)

    def _get_moduli(self, p):
        deriv = 0
        Es = self.Es
        vs = self.vs
        Gs = self.Gs
        E = Es * (( 1.36265e-01 - 1.22204e-02*vs) * (p**(1-deriv)) * (1+0*deriv) + \
	     ( 8.57991e-02 + 6.63677e-02*vs) * (p**(2-deriv)) * (1+1*deriv) + \
	     ( 7.39887e-01 - 6.26129e-02*vs) * (p**(3-deriv)) * (1+2*deriv))
        v = ( 3.29529e-01 + 1.86038e-02*vs) * (1-deriv) + \
	     (-1.42155e-01 + 4.57806e-01*vs) * (p**(1-deriv)) * (1+0*deriv) + \
	     (-3.29837e-01 + 5.59823e-02*vs) * (p**(2-deriv)) * (1+1*deriv) + \
	     ( 1.41233e-01 + 4.72695e-01*vs) * (p**(3-deriv)) * (1+2*deriv)
        G = Gs * (( 2.17676e-01 + 7.22515e-02*vs) * (p**(1-deriv)) * (1+0*deriv) + \
	     (-7.63847e-02 + 1.31601e+00*vs) * (p**(2-deriv)) * (1+1*deriv) + \
	     ( 9.11800e-01 - 1.55261e+00*vs) * (p**(3-deriv)) * (1+2*deriv))
    
        return (E, v, G)

    def _get_moduli_deriv(self, p):
        deriv = 1
        Es = self.Es
        vs = self.vs
        Gs = self.Gs
        DE = Es * (( 2.05292e-01 - 3.30265e-02*vs) * (p**(1-deriv)) * (1+0*deriv) + 
	     ( 8.12145e-02 + 2.72431e-01*vs) * (p**(2-deriv)) * (1+1*deriv) +
	     ( 6.49737e-01 - 2.42374e-01*vs) * (p**(3-deriv)) * (1+2*deriv))
        Dv =( 2.47760e-01 + 1.69804e-02*vs) * (1-deriv) + \
	     (-1.59293e-01 + 7.38598e-01*vs) * (p**(1-deriv)) * (1+0*deriv) + \
	     (-1.86279e-01 - 4.83229e-01*vs) * (p**(2-deriv)) * (1+1*deriv) + \
	     ( 9.77457e-02 + 7.26595e-01*vs) * (p**(3-deriv)) * (1+2*deriv)
        DG = Gs * (( 1.63200e-01 + 1.27910e-01*vs) * (p**(1-deriv)) * (1+0*deriv) + 
	     ( 6.00810e-03 + 4.13331e-01*vs) * (p**(2-deriv)) * (1+1*deriv) + 
	     ( 7.22847e-01 - 3.56032e-01*vs) * (p**(3-deriv)) * (1+2*deriv))

        return (DE, Dv, DG)  

class ORCTruss(SIMP):
    def __init__(self, Es, vs):
        super().__init__(Es, vs)

    def _get_moduli(self, p):
        deriv = 0
        Es = self.Es
        vs = self.vs
        Gs = self.Gs
        E = Es * (( 1.34332e-01 - 7.06384e-02*vs) * (p**(1-deriv)) * (1+0*deriv) + \
	     ( 2.59957e-01 + 8.51515e-01*vs) * (p**(2-deriv)) * (1+1*deriv) + \
	     ( 6.53902e-01 - 7.29803e-01*vs) * (p**(3-deriv)) * (1+2*deriv))
        v = ( 3.38525e-01 + 7.04361e-03*vs) * (1-deriv) + \
	     (-4.25721e-01 + 4.14882e-01*vs) * (p**(1-deriv)) * (1+0*deriv) + \
	     (-7.68215e-02 + 5.58948e-01*vs) * (p**(2-deriv)) * (1+1*deriv) + \
	     ( 1.64073e-01 + 3.98374e-02*vs) * (p**(3-deriv)) * (1+2*deriv)
        G = Gs * (( 1.96762e-01 + 1.66705e-01*vs) * (p**(1-deriv)) * (1+0*deriv) + \
	     ( 1.30938e-01 + 1.72565e-01*vs) * (p**(2-deriv)) * (1+1*deriv) + \
	     ( 6.45455e-01 - 2.87424e-01*vs) * (p**(3-deriv)) * (1+2*deriv))
    
        return (E, v, G)

    def _get_moduli_deriv(self, p):
        deriv = 1
        Es = self.Es
        vs = self.vs
        Gs = self.Gs
        DE = Es * (( 1.34332e-01 - 7.06384e-02*vs) * (p**(1-deriv)) * (1+0*deriv) + \
	     ( 2.59957e-01 + 8.51515e-01*vs) * (p**(2-deriv)) * (1+1*deriv) + \
	     ( 6.53902e-01 - 7.29803e-01*vs) * (p**(3-deriv)) * (1+2*deriv))
        Dv = ( 3.38525e-01 + 7.04361e-03*vs) * (1-deriv) + \
	     (-4.25721e-01 + 4.14882e-01*vs) * (p**(1-deriv)) * (1+0*deriv) + \
	     (-7.68215e-02 + 5.58948e-01*vs) * (p**(2-deriv)) * (1+1*deriv) + \
	     ( 1.64073e-01 + 3.98374e-02*vs) * (p**(3-deriv)) * (1+2*deriv)
        DG = Gs * (( 1.96762e-01 + 1.66705e-01*vs) * (p**(1-deriv)) * (1+0*deriv) + \
	     ( 1.30938e-01 + 1.72565e-01*vs) * (p**(2-deriv)) * (1+1*deriv) + \
	     ( 6.45455e-01 - 2.87424e-01*vs) * (p**(3-deriv)) * (1+2*deriv))

        return (DE, Dv, DG)  

class Bound(SIMP):
    def __init__(self, Es, vs):
        super().__init__(Es, vs)

    def _get_moduli(self, p):
        vs = self.vs
        Gs = self.Gs

        Ks = 1.0 / (3*(1-2*vs))
        K = Ks + (1-p) / ( -1.0/Ks + p/(Ks + (4.0*Gs)/3.0) )
        G = Gs + (1-p) / ( -1.0/Gs + (2.0*p*(Ks+2.0*Gs)) / (5.0*Gs*(Ks+(4.0*Gs)/3.0)) )
        E = 9*K*G/(3*K+G)
        v = (3*K-2*G) / (2*(3*K+G))
    
        return (E, v, G)

    def _get_moduli_deriv(self, p):
        vs = self.vs
        Gs = self.Gs
        Ks = 1.0 / (3*(1-2*vs))
        K = Ks + (1-p) / ( -1.0/Ks + p/(Ks + (4.0*Gs)/3.0) )
        G = Gs + (1-p) / ( -1.0/Gs + (2.0*p*(Ks+2.0*Gs)) / (5.0*Gs*(Ks+(4.0*Gs)/3.0)) )
        DK = (p - 1)/(((4*Gs)/3 + Ks)*(p/((4*Gs)/3 + Ks) - 1/Ks)**2) -\
         1/(p/((4*Gs)/3 + Ks) - 1/Ks)

        DG = 1/(1/Gs - (2*p*(2*Gs + Ks))/(5*Gs*((4*Gs)/3 + Ks))) + \
        (2*(2*Gs + Ks)*(p - 1))/(5*Gs*((4*Gs)/3 + Ks)*(1/Gs - \
        (2*p*(2*Gs + Ks))/(5*Gs*((4*Gs)/3 + Ks)))**2)

        DE = (9*(3*K+G)*(DK*G+K*DG) - 9*K*G*(3*DK+DG) ) / (3*K+G)**2
        Dv = (2*(3*K+G)*(3*DK-2*DG) - 2*(3*K-2*G)*(3*DK+DG) ) / (2*(3*K+G))**2
   
    
        return (DE, Dv, DG)

MictroSturctDict = {'iso':IsoTruss,
                    'simp':SIMP,
                    'octet':OctetTruss,
                    'orc':ORCTruss,
                    'bound':Bound}


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 2,sharey=True)
    Es = 0.3
    v = [0.2, 0.3, 0.4]
    Gs = [Es / (2*(1+vs)) for vs in v]
    micros = [IsoTruss, OctetTruss, ORCTruss]
    labels = ['ISO', 'OCTET', 'ORC']
    colors = ['red', 'green', 'blue']
    linestyles = ['solid', 'dashed', 'dashdot']
    x = np.linspace(0, 1, 101)
    
    print(micros[0](Es,v[0])._get_moduli(x)[0])



    for i in range(2):
        for j in range(2):
            if i == 1 and j == 1: break
            for k in range(len(v)):
                for l in range(len(micros)):

                    if i == 0 and j == 0:
                        y = micros[l](Es, v[k])._get_moduli(x)[0] / Es
                    elif i == 0 and j == 1:
                        y = micros[l](Es, v[k])._get_moduli(x)[1]
                    else:
                        y = micros[l](Es, v[k])._get_moduli(x)[2] / Gs[k]
                   
                    axs[i,j].plot(x, y, linestyle=linestyles[k],color=colors[l],
                    label=labels[l]+f'v={v[k]}')
            axs[i,j].legend()
    plt.show()
