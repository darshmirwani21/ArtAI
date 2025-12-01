// DOM Elements
const fileInput = document.getElementById('fileInput');
const uploadArea = document.getElementById('uploadArea');
const uploadContent = document.getElementById('uploadContent');
const imagePreview = document.getElementById('imagePreview');
const previewImage = document.getElementById('previewImage');
const removeBtn = document.getElementById('removeBtn');
const styleSelect = document.getElementById('styleSelect');
const analyzeBtn = document.getElementById('analyzeBtn');
const resultsSection = document.getElementById('resultsSection');
const closeResults = document.getElementById('closeResults');
const errorMessage = document.getElementById('errorMessage');

let selectedFile = null;

// Upload area click handler
uploadArea.addEventListener('click', () => {
    if (!selectedFile) {
        fileInput.click();
    }
});

// File input change handler
fileInput.addEventListener('change', (e) => {
    handleFile(e.target.files[0]);
    
    analyzeBtn.disabled = true;
});

// Drag and drop handlers
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    
    if (e.dataTransfer.files.length > 0) {
        handleFile(e.dataTransfer.files[0]);
    }
});

// Remove image handler
removeBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    resetUpload();
});

// Analyze button handler
analyzeBtn.addEventListener('click', async () => {
    if (!selectedFile) return;
    
    await analyzeImage();
});

// Close results handler
closeResults.addEventListener('click', () => {
    resultsSection.style.display = 'none';
});

function handleFile(file) {
    if (!file) return;
    
    // Validate file type
    const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/bmp', 'image/webp'];
    if (!validTypes.includes(file.type)) {
        showError('Please upload a valid image file (JPG, PNG, GIF, BMP, or WEBP)');
        return;
    }
    
    // Validate file size (16MB)
    if (file.size > 16 * 1024 * 1024) {
        showError('File size must be less than 16MB');
        return;
    }
    
    selectedFile = file;
    
    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        uploadContent.style.display = 'none';
        imagePreview.style.display = 'block';
        analyzeBtn.disabled = false;
        hideError();
    };
    reader.readAsDataURL(file);
}

function resetUpload() {
    selectedFile = null;
    fileInput.value = '';
    uploadContent.style.display = 'block';
    imagePreview.style.display = 'none';
    analyzeBtn.disabled = true;
    resultsSection.style.display = 'none';
    hideError();
}

async function analyzeImage() {
    if (!selectedFile) return;
    
    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('target_style', styleSelect.value);
    
    // Show loading state
    analyzeBtn.disabled = true;
    const btnText = analyzeBtn.querySelector('.btn-text');
    const btnLoader = analyzeBtn.querySelector('.btn-loader');
    btnText.style.display = 'none';
    btnLoader.style.display = 'inline-block';
    hideError();
    
    try {
        const response = await fetch('/analyze', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'Analysis failed');
        }
        
        displayResults(data.feedback);
        
    } catch (error) {
        showError(error.message || 'An error occurred during analysis');
        console.error('Error:', error);
    } finally {
        // Reset button state
        analyzeBtn.disabled = false;
        btnText.style.display = 'inline';
        btnLoader.style.display = 'none';
    }
}

function displayResults(feedback) {
    // Target style
    document.getElementById('targetStyle').textContent = 
        feedback.target_style.charAt(0).toUpperCase() + feedback.target_style.slice(1);
    
    // Style characteristics
    const characteristicsList = document.getElementById('styleCharacteristics');
    characteristicsList.innerHTML = feedback.style_characteristics
        .map(char => `<li>${char.charAt(0).toUpperCase() + char.slice(1)}</li>`)
        .join('');
    
    // Recommendations
    const recommendationsList = document.getElementById('recommendations');
    recommendationsList.innerHTML = feedback.recommendations
        .map(rec => `<li>${rec}</li>`)
        .join('');
    
    // Color suggestions
    const colorSuggestions = document.getElementById('colorSuggestions');
    colorSuggestions.innerHTML = feedback.color_suggestions
        .map(color => `<span class="tag">${color.charAt(0).toUpperCase() + color.slice(1)}</span>`)
        .join('');
    
    // Technique suggestions
    const techniqueSuggestions = document.getElementById('techniqueSuggestions');
    techniqueSuggestions.innerHTML = feedback.technique_suggestions
        .map(tech => `<span class="tag">${tech.charAt(0).toUpperCase() + tech.slice(1)}</span>`)
        .join('');
    
    // Technical analysis
    const technicalAnalysis = document.getElementById('technicalAnalysis');
    const tech = feedback.technical_analysis.image_characteristics;
    technicalAnalysis.innerHTML = `
        <div class="tech-stat">
            <label>Dimensions</label>
            <value>${tech.dimensions[0]} × ${tech.dimensions[1]} px</value>
        </div>
        <div class="tech-stat">
            <label>Aspect Ratio</label>
            <value>${tech.aspect_ratio.toFixed(2)}</value>
        </div>
        <div class="tech-stat">
            <label>Brightness</label>
            <value>${tech.brightness.toFixed(1)}</value>
        </div>
        <div class="tech-stat">
            <label>Contrast</label>
            <value>${tech.contrast.toFixed(1)}</value>
        </div>
        <div class="tech-stat">
            <label>Texture Complexity</label>
            <value>${tech.texture_complexity.toFixed(3)}</value>
        </div>
    `;
    
    // Show results
    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function showError(message) {
    errorMessage.textContent = message;
    errorMessage.style.display = 'block';
    errorMessage.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function hideError() {
    errorMessage.style.display = 'none';
}

