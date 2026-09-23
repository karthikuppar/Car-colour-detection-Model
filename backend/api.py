import io
import base64
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from backend.src.utils.draw import TrafficAnnotator

app = FastAPI(
    title='Traffic Signal Analytics API',
    description='Computer vision API for vehicle detection, blue car classification, and pedestrian counting.',
    version='1.0.0'
)

# Enable CORS for local React development
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

# Initialize annotator pipeline once at startup
annotator = TrafficAnnotator()

@app.get('/health')
def health_check():
    return {'status': 'healthy', 'service': 'Traffic Vision API'}

@app.post('/api/analyze')
async def analyze_traffic_image(file: UploadFile = File(...)):
    # Validate uploaded content type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail='Uploaded file must be an image.')

    try:
        # Read uploaded image bytes into OpenCV format
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail='Failed to decode image.')

        # Temporary path for annotator or direct pass
        temp_input_path = 'dataset/_temp_api_in.jpg'
        cv2.imwrite(temp_input_path, img)

        # Process through detection + color classification pipeline
        annotated_img, summary = annotator.process_and_annotate(temp_input_path)

        # Encode annotated image to JPEG buffer
        _, buffer = cv2.imencode('.jpg', annotated_img)
        base64_image = base64.b64encode(buffer).decode('utf-8')

        return {
            'success': True,
            'summary': summary,
            'image_base64': f'data:image/jpeg;base64,{base64_image}'
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run('backend.api:app', host='127.0.0.1', port=8000, reload=True)
