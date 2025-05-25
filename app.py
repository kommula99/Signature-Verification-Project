from flask import Flask, request, render_template, jsonify
import cv2
import numpy as np
from signature_verification import SignatureVerification
from PIL import Image
import io
import os

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "default-signature-verification-secret")
verifier = SignatureVerification()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/verify', methods=['POST'])
def verify_signature():
    try:
        if 'signature' not in request.files:
            return jsonify({'error': 'No signature uploaded'}), 400
        
        file = request.files['signature']
        if not file.filename:
            return jsonify({'error': 'Empty file submitted'}), 400
            
        # Validate file type
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
        if not file.filename.lower().rsplit('.', 1)[1] in allowed_extensions:
            return jsonify({'error': 'Invalid file type. Please upload an image file'}), 400
        
        # Read and validate the image
        try:
            img = Image.open(io.BytesIO(file.read()))
            img = np.array(img)
            
            if img.size == 0:
                return jsonify({'error': 'Invalid image data'}), 400
                
        except Exception as e:
            return jsonify({'error': f'Error processing image: {str(e)}'}), 400
        
        # Make prediction with error handling
        try:
            prediction, probability = verifier.predict(img)
            
            result = {
                'prediction': 'Genuine' if prediction == 1 else 'Forged',
                'confidence': float(max(probability))
            }
            
            return jsonify(result)
            
        except Exception as e:
            return jsonify({'error': f'Error during signature verification: {str(e)}'}), 500
            
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
