document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const uploadForm = document.getElementById('uploadForm');
    const previewArea = document.getElementById('previewArea');
    const imagePreview = document.getElementById('imagePreview');
    const removeButton = document.getElementById('removeButton');
    const verifyButton = document.getElementById('verifyButton');
    const loadingIndicator = document.getElementById('loadingIndicator');
    const errorAlert = document.getElementById('errorAlert');
    const errorMessage = document.getElementById('errorMessage');
    const resultsCard = document.getElementById('resultsCard');
    const resultsHeader = document.getElementById('resultsHeader');
    const resultsTitle = document.getElementById('resultsTitle');
    const resultText = document.getElementById('resultText');
    const resultIcon = document.getElementById('resultIcon');
    const confidenceMeter = document.getElementById('confidenceMeter');
    const confidenceExplanation = document.getElementById('confidenceExplanation');

    // Drag and Drop functionality
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, unhighlight, false);
    });

    function highlight() {
        dropZone.classList.add('highlight');
    }

    function unhighlight() {
        dropZone.classList.remove('highlight');
    }

    // Handle file drop
    dropZone.addEventListener('drop', handleDrop, false);

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length) {
            fileInput.files = files;
            handleFiles(files);
        }
    }

    // Handle file selection via input
    fileInput.addEventListener('change', function() {
        if (fileInput.files.length) {
            handleFiles(fileInput.files);
        }
    });

    // Click on drop zone to trigger file input
    dropZone.addEventListener('click', function() {
        fileInput.click();
    });

    // Handle the selected files
    function handleFiles(files) {
        const file = files[0];
        
        // Validate file type
        const validTypes = ['image/jpeg', 'image/png', 'image/gif'];
        if (!validTypes.includes(file.type)) {
            showError('Please upload a valid image file (JPEG, PNG, or GIF).');
            resetUpload();
            return;
        }
        
        // Validate file size (max 5MB)
        if (file.size > 5 * 1024 * 1024) {
            showError('File is too large. Maximum file size is 5MB.');
            resetUpload();
            return;
        }

        // Display image preview
        const reader = new FileReader();
        reader.onload = function(e) {
            imagePreview.src = e.target.result;
            previewArea.style.display = 'block';
            hideError();
            
            // Hide results if showing a new image
            hideResults();
        };
        reader.readAsDataURL(file);
    }

    // Remove uploaded image
    removeButton.addEventListener('click', function() {
        resetUpload();
        hideResults();
    });

    // Reset upload area
    function resetUpload() {
        uploadForm.reset();
        previewArea.style.display = 'none';
        imagePreview.src = '#';
    }

    // Verify signature
    verifyButton.addEventListener('click', verifySignature);

    function verifySignature() {
        if (!fileInput.files.length) {
            showError('Please upload an image first.');
            return;
        }

        // Show loading indicator
        loadingIndicator.style.display = 'block';
        hideError();
        hideResults();
        
        const formData = new FormData();
        formData.append('signature', fileInput.files[0]);

        fetch('/verify', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(data => {
                    throw new Error(data.error || 'Verification failed. Please try again.');
                });
            }
            return response.json();
        })
        .then(data => {
            displayResults(data);
        })
        .catch(error => {
            showError(error.message);
        })
        .finally(() => {
            loadingIndicator.style.display = 'none';
        });
    }

    // Display verification results
    function displayResults(data) {
        resultsCard.classList.remove('d-none');
        
        const isGenuine = data.prediction === 'Genuine';
        const confidencePercentage = Math.round(data.confidence * 100);
        
        // Update UI elements
        resultText.textContent = data.prediction;
        confidenceMeter.style.width = `${confidencePercentage}%`;
        confidenceMeter.textContent = `${confidencePercentage}%`;
        
        // Update confidence explanation
        let explanation = '';
        if (confidencePercentage >= 90) {
            explanation = 'Very high confidence in the result.';
        } else if (confidencePercentage >= 70) {
            explanation = 'Good confidence in the result.';
        } else if (confidencePercentage >= 50) {
            explanation = 'Moderate confidence in the result.';
        } else {
            explanation = 'Low confidence in the result. Consider uploading a clearer image.';
        }
        confidenceExplanation.textContent = explanation;
        
        // Apply styling based on result
        if (isGenuine) {
            resultIcon.className = 'fas fa-check-circle fa-3x';
            resultIcon.parentElement.className = 'result-icon-container genuine-result';
            resultsHeader.className = 'card-header bg-success bg-opacity-25';
            resultsTitle.className = 'mb-0 text-success';
            resultText.className = 'mt-3 text-success';
            confidenceMeter.className = 'progress-bar progress-bar-striped bg-success';
        } else {
            resultIcon.className = 'fas fa-times-circle fa-3x';
            resultIcon.parentElement.className = 'result-icon-container forged-result';
            resultsHeader.className = 'card-header bg-danger bg-opacity-25';
            resultsTitle.className = 'mb-0 text-danger';
            resultText.className = 'mt-3 text-danger';
            confidenceMeter.className = 'progress-bar progress-bar-striped bg-danger';
        }
        
        // Display processing steps if available
        if (data.visualizations) {
            const processingSteps = document.getElementById('processingSteps');
            processingSteps.classList.remove('d-none');
            
            // Set the images for each processing step
            document.getElementById('originalImage').src = data.visualizations.original;
            document.getElementById('grayscaleImage').src = data.visualizations.grayscale;
            document.getElementById('binaryImage').src = data.visualizations.binary;
            document.getElementById('resizedImage').src = data.visualizations.resized;
            document.getElementById('hogImage').src = data.visualizations.hog_image;
            document.getElementById('densityImage').src = data.visualizations.density_image;
        }
    }

    // Hide results
    function hideResults() {
        resultsCard.classList.add('d-none');
    }

    // Error handling
    function showError(message) {
        errorAlert.style.display = 'block';
        errorMessage.textContent = message;
        loadingIndicator.style.display = 'none';
    }

    function hideError() {
        errorAlert.style.display = 'none';
    }
});