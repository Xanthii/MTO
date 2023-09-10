import numpy as np
from Parser import tpd_file2dict
from scipy.sparse.linalg import splu
MAX_ITERS = 250

SOLID, VOID = 1.000, 0.001 #  Upper and lower bound value for design variables
KDATUM = 0.1 #  Reference stiffness value of springs for mechanism synthesis

# Constants for exponential approximation:
A_LOW = -3 #  Lower restriction on 'a' for exponential approximation
A_UPP = -1e-5 #  Upper restriction on 'a' for exponential approximation

class Topology:
    def __init__(self, Es=1.0, vs=0.3, minVF=0.001,maxVF=1.,change=1) -> None:
        self.Es = Es
        self.vs = vs
        self.Gs = Es / (2*(1+vs))
        self.minVF=minVF
        self.maxVF=maxVF
        self.change = change
    # ======================
    # === Public methods ===
    # ======================
    def load_tpd_file(self, fname):
        self.tpdfname = fname 
        self.topydict = tpd_file2dict(fname)

    def set_top_params(self):
        '''
        firstly, consider the general 3d case
        '''
        if not self.topydict:
            raise Exception('You must first load a TPD file!')
        self.probtype = self.topydict['PROB_TYPE'] #  Problem type
        self.probname = self.topydict.get('PROB_NAME', '') #  Problem name
        self.volfrac = self.topydict['VOL_FRAC'] #  Volume fraction
        self.filtrad = self.topydict['FILT_RAD'] #  Filter radius
        self.p = self.topydict['P_FAC'] #  'Standard' penalisation factor
        self.dofpn = self.topydict['DOF_PN'] #  DOF per node
        self.e2sdofmapi = self.topydict['E2SDOFMAPI'] #  Elem to structdof map
        self.nelx = self.topydict['NUM_ELEM_X'] #  Number of elements in X
        self.nely = self.topydict['NUM_ELEM_Y'] #  Number of elements in Y
        self.nelz = self.topydict['NUM_ELEM_Z'] #  Number of elements in Z
        self.fixdof = self.topydict['FIX_DOF'] #  Fixed dof vector
        self.loaddof = self.topydict['LOAD_DOF'] #  Loaded dof vector
        self.loadval = self.topydict['LOAD_VAL'] #  Loaded dof values
        # self.Ke = self.topydict['ELEM_K'] #  Element stiffness matrix
        self.K = self.topydict['K'] #  Global stiffness matrix

        # Check for either one of the following two, will take NUM_ITER if both
        # are specified.
        try:
            self.numiter = self.topydict['NUM_ITER'] #  Number of iterations
            
        except KeyError:
            self.chgstop = self.topydict['CHG_STOP'] #  Change stop criteria
            self.numiter = MAX_ITERS

        # All DOF vector and design variables arrays:
        if self.dofpn == 1:
            if self.nelz == 0: #  *had to this
                self.e2sdofmapi = self.e2sdofmapi[0:4]
                self.alldof = np.arange(self.dofpn * (self.nelx + 1) * \
                    (self.nely + 1))
                self.desvars = np.zeros((self.nely, self.nelx)) + self.volfrac
            else:
                self.alldof = np.arange(self.dofpn * (self.nelx + 1) * \
                    (self.nely + 1) * (self.nelz + 1))
                self.desvars = np.zeros((self.nelz, self.nely, self.nelx)) + \
                    self.volfrac
        elif self.dofpn == 2:
            self.alldof = np.arange(self.dofpn * (self.nelx + 1) * (self.nely + 1))
            self.desvars = np.zeros((self.nely, self.nelx)) + self.volfrac
        else:
            self.alldof = np.arange(self.dofpn * (self.nelx + 1) *\
                (self.nely + 1) * (self.nelz + 1))
            self.desvars = np.zeros((self.nelz, self.nely, self.nelx)) + \
                self.volfrac
        
        self.df = np.zeros_like(self.desvars) #  Derivatives of obj. func. (array)
        self.freedof = np.setdiff1d(self.alldof, self.fixdof) #  Free DOF vector
        self.r = np.zeros_like(self.alldof).astype(float) #  Load vector
        self.r[self.loaddof] = self.loadval #  Assign load values at loaded dof
        self.rfree = self.r[self.freedof] #  Modified load vector (free dof)
        self.d = np.zeros_like(self.r) #  Displacement vector
        self.dfree = np.zeros_like(self.rfree) #  Modified load vector (free dof)
        # Determine which rows and columns must be deleted from global K:
        self._rcfixed = np.where(np.in1d(self.alldof, self.fixdof), 0, 1)

    def fea(self):
        Kfree = self._updateK(self.K.copy())
        Kfree = Kfree.tocsc()
        lu = splu(Kfree)
        self.dfree = lu.solve(self.rfree)
        self.d[self.freedof] = self.dfree   
   

    def sens_analysis(self):
        self.objfval  = 0.0 #  Objective function value
        for elz in range(self.nelz):
            for elx in range(self.nelx):
                for ely in range(self.nely):
                    e2sdofmap = self.e2sdofmapi + self.dofpn *\
                                (ely + elx * (self.nely + 1) + elz *\
                                (self.nelx + 1) * (self.nely + 1))
                    if self.probtype == 'comp' or self.probtype == 'mech':
                        Ke = self._get_KE(elx,ely,elz)
                        DKe = self._get_KE_deriv(elx,ely,elz)
                        de = self.d[e2sdofmap]
                 
                        self.objfval += de @ Ke @ de
                        self.df[elz, ely, elx] = -de @ DKe @ de
    
    def filter_sens_sigmund(self):
        """
        Filter the design sensitivities using Sigmund's heuristic approach.
        Return the filtered sensitivities.

        EXAMPLES:
            >>> t.filter_sens_sigmund()

        See also: sens_analysis

        """
        tmp = np.zeros_like(self.df)
        rmin = int(np.floor(self.filtrad))
    
        rmin3 = rmin
        U, V, W = np.indices((self.nelx, self.nely, self.nelz))
        for i in range(self.nelx):
            umin, umax = np.maximum(i - rmin - 1, 0),\
                            np.minimum(i + rmin + 2, self.nelx + 1)
            for j in range(self.nely):
                vmin, vmax = np.maximum(j - rmin - 1, 0),\
                                np.minimum(j + rmin + 2, self.nely + 1)
                for k in range(self.nelz):
                    wmin, wmax = np.maximum(k - rmin3 - 1, 0),\
                                    np.minimum(k + rmin3 + 2, self.nelz + 1)
                    u = U[umin:umax, vmin:vmax, wmin:wmax]
                    v = V[umin:umax, vmin:vmax, wmin:wmax]
                    w = W[umin:umax, vmin:vmax, wmin:wmax]
                    dist = self.filtrad - np.sqrt((i - u) ** 2 + (j - v) **\
                            2 + (k - w) ** 2)
                    sumnumr = (np.maximum(0, dist) * self.desvars[w, v, u] *\
                                self.df[w, v, u]).sum()
                    sumconv = np.maximum(0, dist).sum()
                    tmp[k, j, i] = sumnumr/(sumconv *\
                    self.desvars[k, j, i])

        self.df = tmp

    def update_desvars_oc(self):
        """
        Update the design variables by means of OC-like or equivalently SAO
        method, using the filtered sensitivities; return the updated design
        variables.

        EXAMPLES:
            >>> t.update_desvars_oc()

        See also: sens_analysis, filter_sens_sigmund

        """
        self.desvarsold = self.desvars.copy()
        minVF = self.minVF
        maxVF = self.maxVF
        l1 = 0
        l2 = 100000
        move = 0.2
        while (l2 - l1 > 1e-4):
            lmid = 0.5*(l2 + l1)
            xnew = np.maximum(minVF, np.maximum(self.desvars-move, np.minimum(maxVF, np.minimum(self.desvars+move,self.desvars*np.sqrt(-self.df/lmid)))))
 
            
            if np.sum(xnew) - self.volfrac * self.nelx * self.nely * self.nelz > 0:
                l1 = lmid
            else:
                l2 = lmid

        self.desvars = xnew.copy()
        self.change = (np.abs(self.desvars - self.desvarsold)).max()

        

    def _updateK(self, K):
        """
        Update the global stiffness matrix by looking at each element's
        contribution i.t.o. design domain density and the penalisation factor.
        Return unconstrained stiffness matrix.

        """   
        for elz in range(self.nelz):
            for elx in range(self.nelx):
                for ely in range(self.nely):
                    e2sdofmap = self.e2sdofmapi + self.dofpn *\
                                (ely + elx * (self.nely + 1) + elz *\
                                (self.nelx + 1) * (self.nely + 1))

                    updatedKe = self._get_KE(elx,ely,elz)
                    K[np.ix_(e2sdofmap, e2sdofmap)] += updatedKe.copy()
        return K[np.ix_(self.freedof, self.freedof)]

    def _get_KE(self,elx,ely,elz):
        KE = np.zeros((24, 24))
        CE = self._get_CE(elx,ely,elz)
        GN_x = np.array([-1 / np.sqrt(3), 1 / np.sqrt(3)])
        GN_y = GN_x.copy()
        GN_z = GN_x.copy()
        GaussWeigh=[1, 1]
        dN = np.zeros((9, 24))
        for i in range(len(GN_x)):
            for j in range(len(GN_y)):
                for k in range(len(GN_z)):
                    x = GN_x[i]
                    y = GN_y[j]
                    z = GN_z[k]
                    dNx = 1/8*np.array([-(1-y)*(1-z),  (1-y)*(1-z),  (1+y)*(1-z), -(1+y)*(1-z), -(1-y)*(1+z),  (1-y)*(1+z),  (1+y)*(1+z), -(1+y)*(1+z)])
                    dNy = 1/8*np.array([-(1-x)*(1-z), -(1+x)*(1-z),  (1+x)*(1-z),  (1-x)*(1-z), -(1-x)*(1+z), -(1+x)*(1+z),  (1+x)*(1+z),  (1-x)*(1+z)])
                    dNz = 1/8*np.array([-(1-x)*(1-y), -(1+x)*(1-y), -(1+x)*(1+y), -(1-x)*(1+y),  (1-x)*(1-y),  (1+x)*(1-y),  (1+x)*(1+y),  (1-x)*(1+y)])
            
                    dN[0,0:24:3] = dNx
                    dN[1,0:24:3] = dNy
                    dN[2,0:24:3] = dNz

                    dN[3,1:24:3] = dNx
                    dN[4,1:24:3] = dNy
                    dN[5,1:24:3] = dNz

                    dN[6,2:24:3] = dNx
                    dN[7,2:24:3] = dNy
                    dN[8,2:24:3] = dNz
                    Be = dN * 2 
                    # Jacobi Matrix J = 0.5*eye()
                    # det(J) = 1/8
                    KE = KE + GaussWeigh[i] * GaussWeigh[j] * GaussWeigh[k] *(Be.T @ CE @ Be) / 8.

        return KE
        
    def _get_CE(self,elx,ely,elz):
        p = self.desvars[elz, ely, elx]
        E,v,G = self._iso_moduli(p)
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

    def _iso_moduli(self, p):
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

        G = Gs * (( 1.63200e-01 + 1.27910e-01*vs) * (p**(1-deriv)) * (1+0*deriv) + \
	     ( 6.00810e-03 + 4.13331e-01*vs) * (p**(2-deriv)) * (1+1*deriv) + \
	     ( 7.22847e-01 - 3.56032e-01*vs) * (p**(3-deriv)) * (1+2*deriv))
        
        return (E, v, G)

################# Deriv
    def _get_KE_deriv(self,elx,ely,elz):
        KE = np.zeros((24, 24))
        CE = self._get_CE_deriv(elx,ely,elz)
        GN_x = np.array([-1 / np.sqrt(3), 1 / np.sqrt(3)])
        GN_y = GN_x.copy()
        GN_z = GN_x.copy()
        GaussWeigh=[1, 1]
        dN = np.zeros((9, 24))
        for i in range(len(GN_x)):
            for j in range(len(GN_y)):
                for k in range(len(GN_z)):
                    x = GN_x[i]
                    y = GN_y[j]
                    z = GN_z[k]
                    dNx = 1/8*np.array([-(1-y)*(1-z),  (1-y)*(1-z),  (1+y)*(1-z), -(1+y)*(1-z), -(1-y)*(1+z),  (1-y)*(1+z),  (1+y)*(1+z), -(1+y)*(1+z)])
                    dNy = 1/8*np.array([-(1-x)*(1-z), -(1+x)*(1-z),  (1+x)*(1-z),  (1-x)*(1-z), -(1-x)*(1+z), -(1+x)*(1+z),  (1+x)*(1+z),  (1-x)*(1+z)])
                    dNz = 1/8*np.array([-(1-x)*(1-y), -(1+x)*(1-y), -(1+x)*(1+y), -(1-x)*(1+y),  (1-x)*(1-y),  (1+x)*(1-y),  (1+x)*(1+y),  (1-x)*(1+y)])
            
                    dN[0,0:24:3] = dNx
                    dN[1,0:24:3] = dNy
                    dN[2,0:24:3] = dNz

                    dN[3,1:24:3] = dNx
                    dN[4,1:24:3] = dNy
                    dN[5,1:24:3] = dNz

                    dN[6,2:24:3] = dNx
                    dN[5,2:24:3] = dNy
                    dN[8,2:24:3] = dNz
                    Be = dN
                    KE = KE + GaussWeigh[i] * GaussWeigh[j] * GaussWeigh[k] *(Be.T @ CE @ Be)
        return KE
        
    def _get_CE_deriv(self,elx,ely,elz):
        p = self.desvars[elz, ely, elx]
        E,v,G = self._iso_moduli(p)
        DE,Dv,DG = self._iso_moduli_deriv(p)
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
        CE = np.array(CE).reshape((9,9))
        return CE

    def _iso_moduli_deriv(self, p):
        deriv = 1
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

        G = Gs * (( 1.63200e-01 + 1.27910e-01*vs) * (p**(1-deriv)) * (1+0*deriv) + \
	     ( 6.00810e-03 + 4.13331e-01*vs) * (p**(2-deriv)) * (1+1*deriv) + \
	     ( 7.22847e-01 - 3.56032e-01*vs) * (p**(3-deriv)) * (1+2*deriv))
        
        return (E, v, G)
    
 


if __name__ == '__main__':
    t = Topology()
    t.load_tpd_file('mmb_beam_2d_reci.tpd')
    t.set_top_params()
    t.fea()
    print(t)
