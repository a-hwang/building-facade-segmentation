import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def build_model(model_name, num_classes):
    """
    Build the object detection model based on the given model_name and number of classes.
    """
    # Get the constructor from torchvision.models.detection using the model name.
    model_fn = getattr(torchvision.models.detection, model_name)
    model = model_fn(weights="DEFAULT")

    # For models that contain ROI heads (i.e., they have a box_predictor)
    # we modify the predictor to handle our number of classes.
    if model_name in {
        "maskrcnn_resnet50_fpn", 
        "fasterrcnn_resnet50_fpn", 
        "fasterrcnn_resnet50_fpn_v2", 
        "fasterrcnn_mobilenet_v3_large_fpn"
    }:
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model