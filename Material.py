class Material:
    def __init__(self, Es, vs) -> None:
        self.Es = Es
        self.vs = vs
        self.Gs = Es / (2 * (1+vs))
