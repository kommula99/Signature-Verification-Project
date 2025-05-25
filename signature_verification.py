import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
import matplotlib.pyplot as plt
import os
import base64
from io import BytesIO

class SignatureVerification:
    def __init__(self):
        self.model = SVC(kernel='rbf', probability=True)
        self.scaler = StandardScaler()
        
        # Initialize with synthetic signature-like data
        np.random.seed(42)
        n_samples = 200  # Increased sample size
        image_height, image_width = 100, 200
        
        # Generate synthetic signature-like images and features
        features_list = []
        for _ in range(n_samples):
            # Create a blank image
            img = np.ones((image_height, image_width)) * 255
            
            # Generate random stroke-like patterns
            n_strokes = np.random.randint(3, 8)
            for _ in range(n_strokes):
                # Random stroke parameters
                start_x = np.random.randint(0, image_width)
                start_y = np.random.randint(0, image_height)
                length = np.random.randint(20, 100)
                thickness = np.random.randint(2, 5)
                
                # Generate stroke points
                x = np.clip(start_x + np.cumsum(np.random.randn(length) * 3), 0, image_width-1)
                y = np.clip(start_y + np.cumsum(np.random.randn(length) * 2), 0, image_height-1)
                
                # Draw the stroke
                for i in range(len(x)-1):
                    cv2.line(img, (int(x[i]), int(y[i])), (int(x[i+1]), int(y[i+1])), 0, thickness)
            
            # Extract HOG features
            hog_features = hog(img, orientations=9, pixels_per_cell=(8, 8),
                               cells_per_block=(2, 2), visualize=False)
            
            # Calculate pixel density
            pixel_density = np.sum(img == 0) / (img.shape[0] * img.shape[1])
            
            # Combine features
            feature_vector = np.concatenate([hog_features, [pixel_density]])
            features_list.append(feature_vector)
        
        # Convert to numpy array
        X = np.array(features_list)
        
        # Create balanced labels (50% genuine, 50% forged)
        y = np.concatenate([np.ones(n_samples//2), np.zeros(n_samples//2)])
        
        # Train the model with synthetic data
        self.train(X, y)
        
    def preprocess_image(self, image):
        """Preprocess image for feature extraction and visualization"""
        # Store original image
        if len(image.shape) == 3:
            # Convert to RGB for visualization
            original = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        else:
            # If already grayscale, make a copy
            original = image.copy()
            
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Apply Otsu's binarization
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Resize image
        resized = cv2.resize(binary, (200, 100))
        
        # Store all processing steps as images
        steps = {
            'original': self._convert_image_to_base64(original),
            'grayscale': self._convert_image_to_base64(gray),
            'binary': self._convert_image_to_base64(binary),
            'resized': self._convert_image_to_base64(resized)
        }
        
        return resized, steps
    
    def _convert_image_to_base64(self, image):
        """Convert an image to base64 for visualization in HTML"""
        # Make sure we have a valid image
        if image is None or image.size == 0:
            return None
            
        # Convert grayscale to RGB if needed (for visualization)
        if len(image.shape) == 2:
            image_for_display = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image_for_display = image.copy()
            
        # Encode image to base64
        _, buffer = cv2.imencode('.png', image_for_display)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return f"data:image/png;base64,{image_base64}"
    
    def extract_features(self, image):
        """Extract features and create visualizations"""
        # Calculate HOG features with visualization
        hog_features, hog_image = hog(image, orientations=9, pixels_per_cell=(8, 8),
                                     cells_per_block=(2, 2), visualize=True)
        
        # Scale HOG image for visualization
        hog_image_normalized = (hog_image - hog_image.min()) / (hog_image.max() - hog_image.min())
        hog_image_uint8 = (hog_image_normalized * 255).astype(np.uint8)
        
        # Calculate pixel density
        pixel_density = np.sum(image == 0) / (image.shape[0] * image.shape[1])
        
        # Create a pixel density visualization (highlight black pixels)
        density_image = np.zeros_like(image)
        density_image[image == 0] = 255
        
        # Combine features
        features = np.concatenate([hog_features, [pixel_density]])
        
        # Store visualizations
        visualizations = {
            'hog_image': self._convert_image_to_base64(hog_image_uint8),
            'density_image': self._convert_image_to_base64(density_image)
        }
        
        return features, visualizations
    
    def train(self, X, y):
        """Train the model with feature vectors and labels"""
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train the model
        self.model.fit(X_train_scaled, y_train)
        
        # Calculate accuracy
        train_accuracy = self.model.score(X_train_scaled, y_train)
        test_accuracy = self.model.score(X_test_scaled, y_test)
        
        return train_accuracy, test_accuracy
    
    def predict(self, image):
        """Predict if a signature is genuine or forged, with visualizations"""
        try:
            if image is None:
                raise ValueError("Input image is None")
                
            if not isinstance(image, np.ndarray):
                raise ValueError("Input must be a numpy array")
                
            # Validate image dimensions
            if len(image.shape) not in [2, 3]:
                raise ValueError("Invalid image dimensions")
                
            # Preprocess the image
            try:
                processed_image, preprocessing_steps = self.preprocess_image(image)
            except Exception as e:
                raise ValueError(f"Error in image preprocessing: {str(e)}")
            
            # Extract features
            try:
                features, feature_visualizations = self.extract_features(processed_image)
            except Exception as e:
                raise ValueError(f"Error in feature extraction: {str(e)}")
            
            # Validate feature extraction
            if features is None or len(features) == 0:
                raise ValueError("Feature extraction failed")
            
            # Check if model is trained
            if not hasattr(self.model, 'fit') or not hasattr(self.scaler, 'transform'):
                raise ValueError("Model not trained. Please train the model first.")
            
            # Scale features
            try:
                features_scaled = self.scaler.transform(features.reshape(1, -1))
            except Exception as e:
                raise ValueError(f"Error in feature scaling: {str(e)}")
            
            # Make prediction
            try:
                prediction = self.model.predict(features_scaled)
                probability = self.model.predict_proba(features_scaled)
                
                # Combine all visualizations
                visualizations = {}
                visualizations.update(preprocessing_steps)
                visualizations.update(feature_visualizations)
                
                return prediction[0], probability[0], visualizations
            except Exception as e:
                raise ValueError(f"Error in prediction: {str(e)}")
                
        except Exception as e:
            raise ValueError(f"Signature verification failed: {str(e)}")