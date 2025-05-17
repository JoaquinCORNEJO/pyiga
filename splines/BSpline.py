class Curve:
    def __init__(self):
        self.degree:int = 0
        self.ctrlpts:list = []
        self.knotvector:list = []

class Surface:
    def __init__(self):
        self.degree_u: int = 0
        self.degree_v:int = 0
        self.ctrlpts:list = []
        self.knotvector_u:list = []
        self.knotvector_v:list = []
        self.ctrlpts_size_u:int = 0
        self.ctrlpts_size_v:int = 0

    def set_ctrlpts(self, ctrlpts, ctrlpts_size_u, ctrlpts_size_v):
        assert self.ctrlpts_size_u == ctrlpts_size_u, "Size problem"
        assert self.ctrlpts_size_v == ctrlpts_size_v, "Size problem"
        self.ctrlpts = ctrlpts
        pass

class Volume:
    def __init__(self):
        self.degree_u: int = 0
        self.degree_v:int = 0
        self.degree_w:int = 0
        self.ctrlpts:list = []
        self.knotvector_u:list = []
        self.knotvector_v:list = []
        self.knotvector_w:list = []
        self.ctrlpts_size_u:int = 0
        self.ctrlpts_size_v:int = 0
        self.ctrlpts_size_w:int = 0

    def set_ctrlpts(self, ctrlpts, ctrlpts_size_u, ctrlpts_size_v, ctrlpts_size_w):
        assert self.ctrlpts_size_u == ctrlpts_size_u, "Size problem"
        assert self.ctrlpts_size_v == ctrlpts_size_v, "Size problem"
        assert self.ctrlpts_size_w == ctrlpts_size_w, "Size problem"
        self.ctrlpts = ctrlpts
        pass
