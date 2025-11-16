from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import io
from model import ArtStyleAnalyzer
import traceback

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize analyzer (singleton pattern for efficiency)
analyzer = None

def get_analyzer():
    """Get or create the analyzer instance."""
    global analyzer
    if analyzer is None:
        analyzer = ArtStyleAnalyzer()
    return analyzer

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze uploaded painting and return style feedback."""
    try:
        # Check if file is in request
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        target_style = request.form.get('target_style', 'impressionist')
        
        # Validate file
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload an image file.'}), 400
        
        # Validate target style
        valid_styles = ['art_nouveau', 'cubism', 'expressionism', 'fauvism', 
                       'futurism', 'impressionism', 'neoclassicism', 'surrealism']
        if target_style not in valid_styles:
            return jsonify({'error': f'Invalid style. Must be one of: {", ".join(valid_styles)}'}), 400
        
        # Read file into memory
        file_stream = io.BytesIO(file.read())
        file_stream.seek(0)  # Reset stream position
        
        # Get analyzer and process
        analyzer = get_analyzer()
        feedback = analyzer.generate_style_feedback(file_stream, target_style)
        
        # Convert numpy types to native Python types for JSON serialization
        tech_analysis = feedback['technical_analysis']['image_characteristics']
        feedback['technical_analysis']['image_characteristics'] = {
            'dimensions': list(tech_analysis['dimensions']),
            'aspect_ratio': float(tech_analysis['aspect_ratio']),
            'brightness': float(tech_analysis['brightness']),
            'contrast': float(tech_analysis['contrast']),
            'texture_complexity': float(tech_analysis['texture_complexity'])
        }
        
        return jsonify({
            'success': True,
            'feedback': feedback
        })
    
    except Exception as e:
        app.logger.error(f"Error during analysis: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/styles', methods=['GET'])
def get_styles():
    """Get available art styles."""
    analyzer = get_analyzer()
    styles = {}
    for style_name, style_info in analyzer.art_styles.items():
        styles[style_name] = {
            'characteristics': style_info['characteristics'],
            'colors': style_info['colors'],
            'technique': style_info['technique']
        }
    return jsonify({'styles': styles})

if __name__ == '__main__':
    print("Starting Art Style Analyzer Web Application...")
    print("Navigate to http://localhost:5000 in your browser")
    app.run(debug=True, host='0.0.0.0', port=5000)

