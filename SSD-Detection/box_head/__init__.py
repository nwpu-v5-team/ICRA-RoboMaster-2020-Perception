import registry
from box_head.box_head import BoxHead
#import box_head.box_losss


def build_boxhead(boxhead):
    boxhead_name = boxhead["name"]
    setting = boxhead["setting"]
    #print(setting)
    #print(registry.BoxHead[boxhead_name])
    #a =registry.BoxHead[boxhead_name](setting)
    return registry.BoxHead[boxhead_name](setting)

