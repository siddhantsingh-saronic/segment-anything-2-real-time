import torch
import cv2
from sam2.build_sam import build_sam2_camera_predictor

sam2_checkpoint = "sam2.1_hiera_tiny.pt"
model_cfg = "sam2.1_hiera_t"
predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint, device="cpu")

cap = cv2.VideoCapture(0)

if_init = False

with torch.inference_mode():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        width, height = frame.shape[:2][::-1]

        if not if_init:
            predictor.load_first_frame(frame)
            if_init = True
            # Add a bounding box prompt - you can modify these coordinates
            bbox = [100, 100, 400, 400]  # [x1, y1, x2, y2]
            _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(
                frame_idx=0, obj_id=1, bbox=bbox
            )
        else:
            out_obj_ids, out_mask_logits = predictor.track(frame)
        
        # Display the frame with masks
        cv2.imshow('SAM2 Real-time', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()