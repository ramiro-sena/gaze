from fastapi import FastAPI, UploadFile, File
from starlette.responses import StreamingResponse
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import io
import base64
import landmark

app = FastAPI()

@app.get("/")
def hello():
    return {"message": "Server is up and running!"}


@app.post("/predict-file")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    data, image = landmark.predict(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # line that fixed it
    _, encoded_img = cv2.imencode('.PNG', image)
    #encoded_img = base64.b64encode(encoded_img)
    headers = (data)
    res = StreamingResponse(io.BytesIO(encoded_img.tobytes()), media_type="image/png")
    return res


