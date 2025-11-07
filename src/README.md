# Art Style Analyzer

A Python program that uses a Vision Transformer (ViT) to analyze uploaded images and provide feedback on how to align them with specific art styles.

## Features

- **Vision Transformer Integration**: Uses pre-trained ViT-B-16 model for feature extraction
- **Multiple Art Styles**: Supports 5 major art movements:
  - Impressionist
  - Cubist
  - Renaissance
  - Expressionist
  - Baroque
- **Comprehensive Analysis**: Provides detailed feedback including:
  - Style-specific recommendations
  - Color suggestions
  - Technique guidance
  - Technical image analysis

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Command Line Interface

```bash
python model.py <image_path> <target_style> [--output feedback.json]
```

**Arguments:**
- `image_path`: Path to the image file to analyze
- `target_style`: Target art style (impressionist, cubist, renaissance, expressionist, baroque)
- `--output`: Optional output file for JSON feedback

**Example:**
```bash
python model.py my_painting.jpg impressionist --output results.json
```

### Programmatic Usage

```python
from model import ArtStyleAnalyzer

# Initialize analyzer
analyzer = ArtStyleAnalyzer()

# Analyze image for specific style
feedback = analyzer.generate_style_feedback('image.jpg', 'impressionist')

# Access recommendations
print(feedback['recommendations'])
print(feedback['color_suggestions'])
print(feedback['technique_suggestions'])
```

### Example Script

Run the example usage script:
```bash
python example_usage.py
```
(Make sure to place an image file named `example_image.jpg` in the project directory)

## Supported Art Styles

### Impressionist
- Characteristics: Loose brushstrokes, soft edges, natural light, outdoor scenes
- Recommended techniques: Visible brush strokes, wet-on-wet, broken color

### Cubist
- Characteristics: Geometric shapes, multiple perspectives, abstract forms
- Recommended techniques: Fragmentation, volumetric analysis, simultaneous perspective

### Renaissance
- Characteristics: Realistic proportions, depth of field, classical composition
- Recommended techniques: Sfumato, chiaroscuro, linear perspective

### Expressionist
- Characteristics: Distorted forms, emotional intensity, bold colors
- Recommended techniques: Thick paint, bold strokes, exaggerated forms

### Baroque
- Characteristics: Dramatic lighting, ornate details, dynamic composition
- Recommended techniques: Tenebrism, foreshortening, dramatic chiaroscuro

## Technical Details

- **Model**: Vision Transformer B-16 (ViT-B-16) pre-trained on ImageNet
- **Input**: Images resized to 224x224 pixels
- **Features**: Extracts high-level visual features from the transformer
- **Analysis**: Combines feature analysis with style-specific heuristics

## Output Format

The program provides structured feedback including:

```json
{
  "target_style": "impressionist",
  "style_characteristics": ["loose brushstrokes", "soft edges", ...],
  "recommendations": ["Increase brushstroke variability...", ...],
  "color_suggestions": ["warm tones", "pastels", ...],
  "technique_suggestions": ["visible brush strokes", ...],
  "technical_analysis": {
    "image_characteristics": {
      "dimensions": [width, height],
      "aspect_ratio": ratio,
      "brightness": value,
      "contrast": value,
      "texture_complexity": value
    }
  }
}
```

## Requirements

- Python 3.7+
- PyTorch 2.0+
- torchvision 0.15+
- Pillow 9.0+
- NumPy 1.21+

## Hardware Requirements

- The program will use CUDA if available, otherwise falls back to CPU
- A GPU with at least 4GB VRAM is recommended for faster processing
- Minimum 8GB RAM recommended for CPU-only operation

## Limitations

- This implementation uses a simplified approach combining pre-trained features with rule-based style analysis
- For production use, consider training the model on style-specific datasets or using fine-tuned models
- Image quality and content significantly affect the analysis results

## Future Enhancements

- Training on style-specific datasets for more accurate analysis
- Support for additional art styles and movements
- Interactive web interface
- Real-time style transfer visualization
- Batch processing capabilities
