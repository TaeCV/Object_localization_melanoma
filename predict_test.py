from mmdet.apis import inference_detector, init_detector, show_result_pyplot

MODEL_CONFIG = './model_config_SmoothL1.py'
CHECKPOINT_PTH = "./mask_rcnn_smoothl1.pth"
IMG_PTH = "./example.jpg"
mask_rcnn_model = init_detector(MODEL_CONFIG, CHECKPOINT_PTH, device='cuda:0')
result = inference_detector(mask_rcnn_model, IMG_PTH)
show_result_pyplot(mask_rcnn_model, IMG_PTH, result, score_thr=0.3)
