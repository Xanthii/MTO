from .parser import tpd_file2dict
class Topology:
    def __init__(self) -> None:
        pass
    def load_tpd_file(self, fname):
        self.tpdfname = fname 
        self.topydict = tpd_file2dict(fname)
    def set_top_params(self):pass
    def fea(self):pass
 
if __name__ == "__main__":
    