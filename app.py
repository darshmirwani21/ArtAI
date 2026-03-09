"""
app.py — STYLO Flask backend
Endpoints:
  GET  /                  → main UI
  POST /analyze           → analyse uploaded painting
  GET  /styles            → list all supported styles + info
  GET  /stats             → per-style usage stats
  POST /feedback/<id>     → submit user rating for an analysis
"""

import io
import os
import time
import traceback

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image

from model import ArtStyleAnalyzer
from database import init_db, get_db, log_analysis, get_recent_analyses, get_style_stats

app = Flask(__name__)
CORS(app)

# ── Config ────────────────────────────────────────────────────────────────────
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024   # 16 MB
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

# ── DB init ───────────────────────────────────────────────────────────────────
init_db()

# ── Analyzer singleton ────────────────────────────────────────────────────────
_analyzer: ArtStyleAnalyzer | None = None

def get_analyzer() -> ArtStyleAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = ArtStyleAnalyzer()
    return _analyzer


def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Accepts multipart/form-data with:
      - file        : image file
      - target_style: one of the supported style keys
    Returns JSON with feedback + classification + DB row id.
    """
    t_start = time.time()

    # ── Validate file ─────────────────────────────────────────────────────────
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    target_style = request.form.get('target_style', 'impressionism').lower()

    if not file.filename:
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Upload JPG, PNG, GIF, BMP, or WEBP.'}), 400

    # ── Validate style ────────────────────────────────────────────────────────
    analyzer = get_analyzer()
    valid_styles = list(analyzer.art_styles.keys())
    if target_style not in valid_styles:
        return jsonify({'error': f'Invalid style. Choose from: {", ".join(valid_styles)}'}), 400

    # ── Read image ────────────────────────────────────────────────────────────
    raw_bytes = file.read()
    file_size_kb = len(raw_bytes) / 1024
    try:
        pil_image = Image.open(io.BytesIO(raw_bytes)).convert('RGB')
    except Exception:
        return jsonify({'error': 'Could not read image file.'}), 400

    # ── Run analysis ──────────────────────────────────────────────────────────
    try:
        feedback = analyzer.generate_style_feedback(pil_image, target_style)
    except Exception as e:
        app.logger.error(traceback.format_exc())
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

    processing_ms = int((time.time() - t_start) * 1000)

    # ── Serialise numpy / tuple types ─────────────────────────────────────────
    chars = feedback['technical_analysis']['image_characteristics']
    feedback['technical_analysis']['image_characteristics'] = {
        'dimensions':         list(chars['dimensions']),
        'aspect_ratio':       float(chars['aspect_ratio']),
        'brightness':         float(chars['brightness']),
        'contrast':           float(chars['contrast']),
        'texture_complexity': float(chars['texture_complexity']),
    }
    # Remove large dominant_colors array from response (kept in DB separately if needed)
    chars.pop('dominant_colors', None)

    # ── Persist to DB ─────────────────────────────────────────────────────────
    analysis_id = None
    try:
        db = next(get_db())
        row = log_analysis(
            db,
            filename      = secure_filename(file.filename or 'upload'),
            file_size_kb  = file_size_kb,
            target_style  = target_style,
            feedback      = feedback,
            processing_ms = processing_ms,
        )
        analysis_id = row.id
    except Exception as e:
        app.logger.warning(f'DB logging failed (non-fatal): {e}')

    return jsonify({
        'success':       True,
        'analysis_id':   analysis_id,
        'processing_ms': processing_ms,
        'feedback':      feedback,
    })


@app.route('/styles', methods=['GET'])
def get_styles():
    """Return all supported styles with their metadata."""
    analyzer = get_analyzer()
    styles = {
        name: {
            'characteristics': info['characteristics'],
            'colors':          info['colors'],
            'technique':       info['technique'],
        }
        for name, info in analyzer.art_styles.items()
    }
    # Flag which styles have a trained classifier behind them
    trained = set(analyzer.trained_styles)
    for name in styles:
        styles[name]['classifier_trained'] = (name in trained)

    return jsonify({'styles': styles})


@app.route('/stats', methods=['GET'])
def stats():
    """Return per-style usage stats from the DB."""
    try:
        db = next(get_db())
        rows = get_style_stats(db)
        return jsonify({
            'stats': [
                {
                    'style':          r.style,
                    'request_count':  r.request_count,
                    'avg_confidence': r.avg_confidence,
                }
                for r in rows
            ]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/feedback/<int:analysis_id>', methods=['POST'])
def submit_feedback(analysis_id: int):
    """
    Let users rate an analysis after seeing results.
    Body: { "rating": 1-5, "comment": "..." }
    """
    data = request.get_json(silent=True) or {}
    rating  = data.get('rating')
    comment = data.get('comment', '')

    if rating is None or not (1 <= int(rating) <= 5):
        return jsonify({'error': 'rating must be an integer 1–5'}), 400

    try:
        from database import SessionLocal, Analysis
        db = SessionLocal()
        row = db.query(Analysis).filter_by(id=analysis_id).first()
        if row is None:
            return jsonify({'error': 'Analysis not found'}), 404
        row.user_rating  = int(rating)
        row.user_comment = str(comment)[:500]
        db.commit()
        db.close()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/history', methods=['GET'])
def history():
    """Return the 20 most recent analyses (for an admin dashboard)."""
    try:
        db = next(get_db())
        rows = get_recent_analyses(db, limit=20)
        return jsonify({'analyses': [r.to_dict() for r in rows]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("Starting STYLO...")
    print("→ http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)