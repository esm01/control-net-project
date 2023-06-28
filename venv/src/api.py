from fastapi import FastAPI
from pydantic import BaseModel
from .pipeline import setup_pipeline, generate_line_image, generate_image
from base64 import b64encode, b64decode
from io import BytesIO
from PIL import Image

app = FastAPI()
pipeline = setup_pipeline()

class ImageRequest(BaseModel):
    prompt: str
    img_base: str

@app.post("/")
async def create_image(img_request: ImageRequest):
    print("a")
    img = Image.open(BytesIO(b64decode(img_request.img_base)))
    print("b")
    line_img = generate_line_image(img)
    print("c")
    output_img = generate_image(line_img, img_request.prompt, pipeline)
    print("d")
    return b64encode(output_img)

