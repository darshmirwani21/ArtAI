# Quick Start Guide

## Getting Started with Art Style Analyzer Web App

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** If you encounter issues installing PyTorch, visit https://pytorch.org/ to get the correct installation command for your system (CPU vs CUDA).

### Step 2: Start the Web Server

```bash
python app.py
```

You should see:
```
Starting Art Style Analyzer Web Application...
Navigate to http://localhost:5000 in your browser
 * Running on http://0.0.0.0:5000
```

### Step 3: Open in Browser

Navigate to: **http://localhost:5000**

### Step 4: Analyze Your Painting

1. **Upload Image:**
   - Drag and drop your painting image onto the upload area
   - Or click the upload area to browse and select a file
   - Supported formats: JPG, PNG, GIF, BMP, WEBP (max 16MB)

2. **Select Target Style:**
   - Choose from the dropdown:
     - Impressionist
     - Cubist
     - Renaissance
     - Expressionist
     - Baroque

3. **Analyze:**
   - Click the "Analyze Painting" button
   - Wait for processing (first time may take longer as the model loads)
   - View your detailed feedback!

### Troubleshooting

**Issue: Port 5000 already in use**
- Change the port in `app.py`: `app.run(port=5001)`

**Issue: Model loading takes too long**
- First load downloads the Vision Transformer model (~330MB)
- Subsequent runs will be faster

**Issue: Out of memory errors**
- The model requires ~2GB RAM minimum
- Close other applications
- For GPU: Requires ~2GB VRAM

**Issue: Import errors**
- Make sure all dependencies are installed: `pip install -r requirements.txt`
- Use Python 3.7 or higher

### Features

✅ Drag & drop file upload
✅ Image preview
✅ Real-time analysis
✅ Detailed style feedback
✅ Technical image analysis
✅ Responsive mobile design
✅ Beautiful modern UI

Enjoy analyzing your art! 

