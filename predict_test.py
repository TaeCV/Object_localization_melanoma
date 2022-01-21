from mmdet.apis import inference_detector, init_detector, show_result_pyplot
import numpy as np


def filter(result, threshold):
    '''
    parameters:
    result - array of size (1,N,5)
    threshold - score threshold
    return:
    list of predictions that score >= threshold {"coordinates":list of 4 coordinates, "score":score} 
    '''
    result = np.array(result).reshape(-1, 5)
    preds = []
    for p in result:
        if p[4] >= threshold:
            preds.append({"coordinates": p[0:4].tolist(), "score": p[4]})

    return preds


MODEL_CONFIG = "./model_config_SmoothL1.py"
CHECKPOINT_PTH = "./epoch_12.pth"
IMG_PTH = "./example.jpg"
mask_rcnn_model = init_detector(MODEL_CONFIG, CHECKPOINT_PTH, device='cpu')
result = inference_detector(mask_rcnn_model, IMG_PTH)
predicted_result = filter(result, 0.3)
print(predicted_result)
#show_result_pyplot(mask_rcnn_model, IMG_PTH, result, score_thr=0.3)
