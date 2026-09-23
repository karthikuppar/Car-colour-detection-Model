import cv2
import base64
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from backend.src.utils.draw import TrafficAnnotator

app = FastAPI(title='Traffic Vision Analytics API', version='1.0')

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

# Initialize detector once at startup
annotator = TrafficAnnotator(yolo_weights='yolov8s.pt')

@app.get('/health')
def health_check():
    return {'status': 'healthy', 'device': 'CUDA / GPU Ready'}

@app.post('/api/analyze')
async def analyze_traffic_image(file: UploadFile = File(...)):
    # Validate file format
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail='Uploaded file must be an image.')

    try:
        # Read file bytes directly into OpenCV image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail='Invalid image data.')

        # Temporarily save to process with annotator pipeline
        temp_input = 'dataset/temp_input.jpg'
        cv2.imwrite(temp_input, img)

        # Run detection, classification, and annotation pipeline
        annotator = TrafficAnnotator()
        annotated_img, summary = annotator.process_and_annotate(temp_input)

        # Encode processed image to base64 JPEG
        _, buffer = cv2.imencode('.jpg', annotated_img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        return {
            'success': True,
            'summary': summary,
            'image_base64': f'data:image/jpeg;base64,{img_base64}'
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run('backend.api:app', host='127.0.0.1', port=8000, reload=True)
