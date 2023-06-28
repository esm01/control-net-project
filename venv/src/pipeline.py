from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.utils import load_image
import torch, cv2
from PIL import Image
import numpy as np
import tensorflow as tf
from .mlsd_utils import pred_lines


def generate_line_image(input_img):
  interpreter = tf.lite.Interpreter(model_path="models/M-LSD_512_large_fp16.tflite")
  interpreter.allocate_tensors()
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  input_img = np.array(input_img)
  output_img = np.zeros(input_img.shape, dtype=np.uint8)

  lines = pred_lines(input_img, interpreter, input_details, output_details, score_thr=0.2, dist_thr=10)
  for line in lines:
    x1, y1, x2, y2 = [int(val) for val in line]
    cv2.line(output_img, (x1, y1),  (x2, y2), (255, 255, 255), 2)

  return Image.fromarray((output_img * 255).astype(np.uint8))

def setup_pipeline():
  controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-mlsd")
  return StableDiffusionControlNetPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", controlnet=controlnet, torch_dtype=torch.float32)

def generate_image(image, prompt, pipeline):
  return pipeline(
    prompt, num_inference_steps=20, image=image
  )
