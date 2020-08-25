

def _register(module_dict, module_name, module):
    module_dict[module_name] = module


class Registry(dict):
    def __init__(self):
        super(Registry, self).__init__()
    def register(self, module_name, module=None):
        def register(fn) :
            _register(self, module_name, fn)
            return fn
        return register



BoxHead = Registry()
BackBone = Registry()
BoxPredictor = Registry()
BoxLossEval = Registry()
LRScheduler = Registry()
PriorBox = Registry()


