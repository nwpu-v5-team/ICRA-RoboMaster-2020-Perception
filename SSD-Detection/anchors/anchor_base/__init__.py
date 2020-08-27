import registry
from anchors.anchor_base.prior_box import PriorBox


def build_priorbox(priorbox):
    priorbox_name = priorbox["name"]
    setting = priorbox["setting"]
    return registry.PriorBox[priorbox_name](setting)