from tensorflow import keras
import uvicorn
import sys
import io
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
import cv2
import pathlib
from fastai.vision.all import load_image

model=keras.models.load_model('new_model.h5')
list = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
path = pathlib.Path(__file__).parent.resolve()

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))

@app.route('/')
async def homepage(request):
    html_file = path / 'view' / 'index.html'
    return HTMLResponse(html_file.open().read())

@app.route('/analyze', methods=['POST'])
async def analyze(request):
    #changes required to execute our model
    img_data = await request.form()
    img_bytes = await (img_data['file'].read())
    img = load_image(io.BytesIO(img_bytes))
    image = cv2.resize(img,(96,96))
    image = image.reshape(-1,96, 96, 3)
    prediction = model.predict(image)
    max = prediction[0][0]
    j = 0
    for i in range(25):
        if prediction[0][i]>max:
            max = prediction[0][i]
            j = i
    return JSONResponse({'result': str(list[j])})
    
if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")
