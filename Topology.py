import Parser
import numpy as np
# from parser import tpd_file2dict
MAX_ITERS = 250

SOLID, VOID = 1.000, 0.001 #  Upper and lower bound value for design variables
KDATUM = 0.1 #  Reference stiffness value of springs for mechanism synthesis

# Constants for exponential approximation:
A_LOW = -3 #  Lower restriction on 'a' for exponential approximation
A_UPP = -1e-5 #  Upper restriction on 'a' for exponential approximation

class Topology:
    def __init__(self) -> None:
        pass


    # ======================
    # === Public methods ===
    # ======================
    def load_tpd_file(self, fname):
        self.tpdfname = fname 
        self.topydict = Parser.tpd_file2dict(fname)

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

        




    def fea(self):pass
 
if __name__ == "__main__":
    t = Topology()
    t.load_tpd_file('mmb_beam_2d_reci.tpd')
    t.set_top_params()
    # topy.optimise(t)