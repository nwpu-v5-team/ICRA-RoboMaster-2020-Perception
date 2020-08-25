import registry
from box_head.box_predicion.box_predict import  BoxPredict


def build_boxpredict(predictor):
    predictor_name = predictor["name"]
    setting = predictor["setting"]
    #print(registry.BoxHead)
    return registry.BoxPredictor[predictor_name](setting)