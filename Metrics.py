import numpy as np

def Metrics(metric_name: str, prediction, ground_truth, threshold=.5) -> float:
    """
    Calculates metrics given prediction and ground truth masks    

    Parameters:
    -----------
    metric_name : str
        Metric to calculate (sensitivity, specificity, dice, accuracy, or iou)
    prediction : array-like
        Predicted values (numpy array, list, or torch tensor)
    ground_truth : array-like
        True values (numpy array, list, or torch tensor)
    threshold : float, default=0.5
        Threshold for converting predictions to binary values
    
    Returns:
    --------
    float
        Calculated metric value
    
    Raises:
    -------
    ValueError if invalid metric_name provided
    """
    
    # Convert mask to numpy if needed
    if isinstance(prediction, list):
        prediction = np.array(prediction)
    elif hasattr(prediction, 'numpy'):  # handles torch.Tensor
        prediction = prediction.cpu().numpy()

    # Convert mask to numpy if needed
    if isinstance(ground_truth, list):
        ground_truth = np.array(ground_truth)
    elif hasattr(ground_truth, 'numpy'):  # handles torch.Tensor
        ground_truth = ground_truth.cpu().numpy()

    # Check for non-binary prediction mask
    if not np.allclose(prediction, prediction.astype(bool), rtol=1e-05, atol=1e-08):
        prediction = prediction >= threshold
    
    TP = np.sum(prediction * ground_truth)
    FP = np.sum(prediction) - TP
    FN = np.sum(ground_truth) - TP
    TN = ground_truth.size - (TP + FP + FN)
    
    match metric_name.casefold():
        case "sensitivity":
            metric = (TP) / (TP + FP)
        case "specificity":
            metric = (TN) / (TN + FP)
        case "dice":
            metric = (2 * TP) / (2*TP + FP + FN)
        case "accuracy":
            metric = (TP + TN) / (TP + TN + FN + FP)
        case "iou":
            metric = TP / (TP + FP + FN)
        case __:
            raise ValueError("Please enter one of the following metrics:\n\
                             Sensitivity\n\
                             Specificity\n\
                             Accuracy\n\
                             IoU\n\
                             Dice\n")

    return metric