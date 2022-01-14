from mmdet.apis import inference_detector, init_detector, show_result_pyplot

MODEL_CONFIG = './model_config_SmoothL1.py'
MODEL_PTH = "./mask_rcnn_smoothl1.pth"
IMG_PTH = "./example.jpg"
Mask_rcnn_model = init_detector(MODEL_CONFIG, MODEL_PTH, device='cuda:0')
result = inference_detector(Mask_rcnn_model, IMG_PTH)
show_result_pyplot(Mask_rcnn_model, IMG_PTH, result, score_thr=0.3)
