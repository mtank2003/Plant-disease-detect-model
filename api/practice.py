from fastapi import File, UploadFile, Request, FastAPI
from fastapi.templating import Jinja2Templates
from fastapi import FastAPI, File, UploadFile, responses, Body, Request
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from io import BytesIO
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from PIL import Image
import tensorflow
from tensorflow.keras.models import load_model
from fastapi.templating import Jinja2Templates


app = FastAPI()
# templates = Jinja2Templates(directory="htmldirectory")

# MODEL_PATH = 'my_h5_model.h5'
MODEL_PATH = 'Model1.h5'
model = load_model(MODEL_PATH)

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]
templates = Jinja2Templates(directory="templates")


# @app.get("/items/", response_class=HTMLResponse)
# async def read_items():
#     return """
#     <html>
#         <head>
#             <title>Some HTML in here</title>
#         </head>
#         <body>
#             <h1>Look ma! HTML!</h1>
#         </body>
#     </html>
#     """


@app.post("/upload")
def upload(file: UploadFile = File(...)):
    try:
        contents = file.file.read()
        with open("uploaded_" + file.filename, "wb") as f:
            f.write(contents)
    except Exception:
        return {"message": "There was an error uploading the file"}
    finally:
        file.file.close()

    return {"message": f"Successfuly uploaded {file.filename}"}


@app.get("/")
def main(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.post("/predict")
# async def predict(
def predict(
    file: UploadFile = File(...)
):

    print()
    print(file)

    image = read_file_as_image(file.read())
    img_batch = np.expand_dims(image, 0)

    predictions = model.predict(img_batch)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return{
        "class": predicted_class,
        "confidence": float(confidence)
    }
    # return   {
    #     'class': predicted_class,
    #     'confidence': float(confidence)
    # }
    # res = {
    #     "class" : predicted_class,
    #     "confidence" : float(confidence)
    # }
    # return templates.TemplateResponse('index.html', res)


    # @app.get("/items/", response_class=HTMLResponse)
    # async def read_items():
    # return """
    # <html>
    #     <head>
    #         <title>Some HTML in here</title>
    #     </head>
    #     <body>
    #         {{ 
    #             {
    #                 'class': predicted_class,
    #                 'confidence': float(confidence)
    #             }
    #         }}
    #     </body>
    # </html>
    # """



    # return templates.TemplateResponse('home.html', {
    #     "class" : predicted_class,
    #     "confidence" : float(confidence)
    # })