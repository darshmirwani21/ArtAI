// ── DOM refs ──────────────────────────────────────────────────────────────────
const fileInput        = document.getElementById('fileInput');
const uploadArea       = document.getElementById('uploadArea');
const uploadContent    = document.getElementById('uploadContent');
const imagePreview     = document.getElementById('imagePreview');
const previewImage     = document.getElementById('previewImage');
const removeBtn        = document.getElementById('removeBtn');
const styleSelect      = document.getElementById('styleSelect');
const analyzeBtn       = document.getElementById('analyzeBtn');
const resultsSection   = document.getElementById('resultsSection');
const closeResults     = document.getElementById('closeResults');
const errorMessage     = document.getElementById('errorMessage');

// Classifier elements
const classifierBanner = document.getElementById('classifierBanner');
const detectedStyle    = document.getElementById('detectedStyle');
const confidenceFill   = document.getElementById('confidenceFill');
const confidenceValue  = document.getElementById('confidenceValue');
const scoreBreakdown   = document.getElementById('scoreBreakdown');
const scoreBars        = document.getElementById('scoreBars');

// Rating elements
const ratingSection    = document.getElementById('ratingSection');
const starRating       = document.getElementById('starRating');
const ratingThanks     = document.getElementById('ratingThanks');

let selectedFile  = null;
let currentAnalysisId = null;

// ── Upload handlers ───────────────────────────────────────────────────────────
uploadArea.addEventListener('click', () => { if (!selectedFile) fileInput.click(); });
fileInput.addEventListener('change', e => handleFile(e.target.files[0]));

uploadArea.addEventListener('dragover', e => { e.preventDefault(); uploadArea.classList.add('dragover'); });
uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('dragover'));
uploadArea.addEventListener('drop', e => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});

removeBtn.addEventListener('click', e => { e.stopPropagation(); resetUpload(); });
analyzeBtn.addEventListener('click', analyzeImage);
closeResults.addEventListener('click', () => { resultsSection.style.display = 'none'; });

// ── Star rating ───────────────────────────────────────────────────────────────
document.querySelectorAll('.star').forEach(star => {
    star.addEventListener('mouseenter', () => highlightStars(+star.dataset.value));
    star.addEventListener('mouseleave', () => highlightStars(0));
    star.addEventListener('click', () => submitRating(+star.dataset.value));
});

function highlightStars(n) {
    document.querySelectorAll('.star').forEach(s => {
        s.classList.toggle('active', +s.dataset.value <= n);
    });
}

async function submitRating(rating) {
    if (!currentAnalysisId) return;
    highlightStars(rating);
    try {
        await fetch(`/feedback/${currentAnalysisId}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ rating })
        });
    } catch (_) {}
    ratingThanks.style.display = 'block';
    starRating.style.pointerEvents = 'none';
}

// ── File handling ─────────────────────────────────────────────────────────────
function handleFile(file) {
    if (!file) return;
    const validTypes = ['image/jpeg','image/jpg','image/png','image/gif','image/bmp','image/webp'];
    if (!validTypes.includes(file.type)) { showError('Please upload a valid image file.'); return; }
    if (file.size > 16 * 1024 * 1024)   { showError('File must be under 16 MB.'); return; }

    selectedFile = file;
    const reader = new FileReader();
    reader.onload = e => {
        previewImage.src = e.target.result;
        uploadContent.style.display = 'none';
        imagePreview.style.display  = 'block';
        analyzeBtn.disabled = false;
        hideError();
    };
    reader.readAsDataURL(file);
}

function resetUpload() {
    selectedFile = null;
    fileInput.value = '';
    uploadContent.style.display = 'block';
    imagePreview.style.display  = 'none';
    analyzeBtn.disabled = true;
    resultsSection.style.display = 'none';
    currentAnalysisId = null;
    hideError();
}

// ── Analysis ──────────────────────────────────────────────────────────────────
async function analyzeImage() {
    if (!selectedFile) return;

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('target_style', styleSelect.value);

    // Loading state
    analyzeBtn.disabled = true;
    analyzeBtn.querySelector('.btn-text').style.display   = 'none';
    analyzeBtn.querySelector('.btn-loader').style.display = 'inline-block';
    hideError();

    try {
        const res  = await fetch('/analyze', { method: 'POST', body: formData });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || 'Analysis failed');

        currentAnalysisId = data.analysis_id;
        displayResults(data.feedback);

    } catch (err) {
        showError(err.message || 'An error occurred during analysis.');
    } finally {
        analyzeBtn.disabled = false;
        analyzeBtn.querySelector('.btn-text').style.display   = 'inline';
        analyzeBtn.querySelector('.btn-loader').style.display = 'none';
    }
}

// ── Display results ───────────────────────────────────────────────────────────
function displayResults(feedback) {
    // Target style label
    const opt = document.querySelector(`#styleSelect option[value="${feedback.target_style}"]`);
    document.getElementById('targetStyle').textContent = opt ? opt.textContent : feedback.target_style;

    // Characteristics
    document.getElementById('styleCharacteristics').innerHTML =
        feedback.style_characteristics
            .map(c => `<li>${cap(c)}</li>`)
            .join('');

    // Recommendations
    document.getElementById('recommendations').innerHTML =
        feedback.recommendations
            .map(r => `<li>${r}</li>`)
            .join('');

    // Colors
    document.getElementById('colorSuggestions').innerHTML =
        feedback.color_suggestions
            .map(c => `<span class="tag">${cap(c)}</span>`)
            .join('');

    // Techniques
    document.getElementById('techniqueSuggestions').innerHTML =
        feedback.technique_suggestions
            .map(t => `<span class="tag">${cap(t)}</span>`)
            .join('');

    // Technical stats
    const tech = feedback.technical_analysis.image_characteristics;
    document.getElementById('technicalAnalysis').innerHTML = `
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
            <label>Texture</label>
            <value>${tech.texture_complexity.toFixed(3)}</value>
        </div>
    `;

    // ── Classifier block ──────────────────────────────────────────────────────
    const clf = feedback.classification;
    if (clf && clf.predicted_style) {
        detectedStyle.textContent   = cap(clf.predicted_style.replace('_', ' '));
        confidenceValue.textContent = `${Math.round(clf.confidence * 100)}%`;
        confidenceFill.style.width  = `${Math.round(clf.confidence * 100)}%`;

        // Colour the confidence bar green → amber → red based on value
        const pct = clf.confidence;
        confidenceFill.style.background =
            pct >= 0.7 ? '#9CAF88' :
            pct >= 0.4 ? '#e6b84a' : '#ef4444';

        classifierBanner.style.display = 'flex';

        // Score breakdown bars
        if (clf.all_scores && Object.keys(clf.all_scores).length) {
            scoreBars.innerHTML = Object.entries(clf.all_scores)
                .map(([style, score]) => `
                    <div class="score-row">
                        <span class="score-name">${cap(style.replace('_', ' '))}</span>
                        <div class="score-track">
                            <div class="score-fill" style="width:${Math.round(score*100)}%;
                                background:${style === clf.predicted_style ? 'var(--primary-color)' : 'var(--border-color)'}">
                            </div>
                        </div>
                        <span class="score-pct">${Math.round(score*100)}%</span>
                    </div>
                `).join('');
            scoreBreakdown.style.display = 'block';
        }
    } else {
        classifierBanner.style.display = 'none';
        scoreBreakdown.style.display   = 'none';
    }

    // Rating
    ratingThanks.style.display = 'none';
    starRating.style.pointerEvents = 'auto';
    highlightStars(0);
    ratingSection.style.display = currentAnalysisId ? 'block' : 'none';

    // Show section
    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function cap(s) { return s.charAt(0).toUpperCase() + s.slice(1); }
function showError(msg) { errorMessage.textContent = msg; errorMessage.style.display = 'block'; }
function hideError()    { errorMessage.style.display = 'none'; }