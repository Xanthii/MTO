from Topology import Topology
import numpy as np 
It = 1
def optimise(topology):
# Optimising function:
    def _optimise(t):
        t.fea()
        t.sens_analysis()
        t.filter_sens_sigmund()
        t.update_desvars_oc()
        vol = np.sum(t.desvars)/(t.nelx * t.nely *t.nelz )
        global It
        print(f' It.:{It}, Obj.:{t.objfval}, Vol.:{vol} ch.:{t.change}')
        It += 1

    try:
        while topology.change > topology.chgstop:
            _optimise(topology)
    except AttributeError:
        for i in range(topology.numiter):
            _optimise(topology)
            
if __name__ == '__main__':
    t = Topology()
    t.load_tpd_file('./examples/mmb_beam_3d_reci.tpd')
    t.set_top_params()
    optimise(t)
