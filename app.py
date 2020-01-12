from starlette.applications import Starlette
from starlette.responses import JSONResponse, HTMLResponse, RedirectResponse
from fastai import *
from fastai.vision import *

import torch
defaults.device = torch.device('cpu')

from pathlib import Path
from io import BytesIO
import sys
import uvicorn
import aiohttp
import os
import base64
from PIL import Image as PILImage


async def get_bytes(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.read()


def encode(img):
    img = (image2np(img.data) * 255).astype('uint8')
    pil_img = PILImage.fromarray(img)
    buff = BytesIO()
    pil_img.save(buff, format="JPEG")
    return base64.b64encode(buff.getvalue()).decode("utf-8")


app = Starlette()

model_path = Path(".")
learner = load_learner(model_path)

@app.route("/upload", methods=["POST"])
async def upload(request):
    data = await request.form()
    bytes = await (data["file"].read())
    return predict_image_from_bytes(bytes)


@app.route("/classify-url", methods=["GET"])
async def classify_url(request):
    bytes = await get_bytes(request.query_params["url"])
    return predict_image_from_bytes(bytes)


def predict_image_from_bytes(bytes):
    img = open_image(BytesIO(bytes))
    pred_class, pred_idx, outputs = learner.predict(img)
    formatted_outputs = ["{:.1f}%".format(value) for value in [x * 100 for x in torch.nn.functional.softmax(outputs, dim=0)]]
    pred_probs = sorted(
        zip(learner.data.classes, map(str, formatted_outputs)),
        key=lambda p: p[1],
        reverse=True
    )
    img_data = encode(img)
    return HTMLResponse(
        """
        <html>
            <body>
                <div style="text-align: center">
                    <p>Prediction: <b>%s</b></p>
                    <p>Confidence: %s</p>
                    <figure class="figure">
                        <img src="data:image/png;base64, %s" class="figure-img img-thumbnail input-image">
                    </figure>
                </div>
           </body>
        </html>
    """ % (pred_class, pred_probs, img_data))

@app.route("/")
def form(request):
    return HTMLResponse(
        """
        <div style="text-align: center">
        <h1>People Boxer</h1>
        <p>Upload a picture of a person and see what kind of stereotype they are.</p><br>
        <form action="/upload" method="post" enctype="multipart/form-data">
            <u>Select image to upload:</u><br><p>
            1. <input type="file" name="file"><br><p>
            2. <input type="submit" value="Upload and analyze image">
        </form>
        <br>
        <strong>OR</strong><br><p>
        <u>Submit a URL:</u>
        <form action="/classify-url" method="get">
            1. <input type="url" name="url" size="60"><br><p>
            2. <input type="submit" value="Fetch and analyze image">
        </form>
        </div>
    """)


@app.route("/form")
def redirect_to_homepage(request):
    return RedirectResponse("/")


if __name__ == "__main__":
    if "serve" in sys.argv:
        port = int(os.environ.get("PORT", 80))
        uvicorn.run(app, host="0.0.0.0", port=port)
