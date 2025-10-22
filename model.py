
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights
from PIL import Image
import numpy as np
import argparse
import os
from typing import List, Dict, Tuple
import json

class ArtStyleAnalyzer:
    """
    A visual transformer-based system for analyzing images and providing 
    feedback on how to align them with specific art styles.
    """
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.is_style_mapping_loaded = False
        
        # Load pre-trained Vision Transformer
        print(f"Loading Vision Transformer on {device}...")
        self.model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        self.model.eval()
        self.model.to(device)
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Art style definitions and their respective characteristics
        self.art_styles = {
            'impressionist': {
                'characteristics': ['loose brushstrokes', 'soft edges', 'natural light', 'outdoor scenes'],
                'colors': ['warm tones', 'pastels', 'natural palette'],
                'technique': ['visible brush strokes', 'wet-on-wet', 'broken color']
            },
            'cubist': {
                'characteristics': ['geometric shapes', 'multiple perspectives', 'abstract forms'],
                'colors': ['muted tones', 'analogous colors', 'limited palette'],
                'technique': ['fragmentation', 'volumetric analysis', 'simultaneous perspective']
            },
            'renaissance': {
                'characteristics': ['realistic proportions', 'depth of field', 'classical composition'],
                'colors': ['earthy tones', 'skin tones', 'natural pigments'],
                'technique': ['sfumato', 'chiaroscuro', 'linear perspective']
            },
            'expressionist': {
                'characteristics': ['distorted forms', 'emotional intensity', 'bold colors'],
                'colors': ['vibrant', 'contrasting', 'non-naturalistic'],
                'technique': ['thick paint', 'bold strokes', 'exaggerated forms']
            },
            'baroque': {
                'characteristics': ['dramatic lighting', 'ornate details', 'dynamic composition'],
                'colors': ['rich golds', 'deep reds', 'contrasting shadows'],
                'technique': ['tenebrism', 'foreshortening', 'dramatic chiaroscuro']
            }
        }
    
    def extract_features(self, image_path: str) -> torch.Tensor:
        """Extract visual features from an uploaded image using the Vision Transformer."""
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Extract features from the last layer before classification
            with torch.no_grad():
                features = self.model.forward_features(input_tensor)
                # Global average pooling
                features = torch.mean(features, dim=1)
            
            return features.cpu()
        
        except Exception as e:
            raise Exception(f"Error processing image: {str(e)}")
    
    def analyze_image_characteristics(self, image_path: str) -> Dict:
        """Analyze visual characteristics of the uploaded image."""
        try:
            image = Image.open(image_path).convert('RGB')
            
            # Basic image analysis
            width, height = image.size
            aspect_ratio = width / height
            
            # Color analysis
            img_array = np.array(image)
            
            # Extract dominant colors
            pixels = img_array.reshape(-1, 3)
            dominant_colors = pixels
            
            # Analyze color properties
            brightness = np.mean(img_array)
            contrast = np.std(img_array)
            
            # Edge detection for brushstroke analysis
            gray = transforms.functional.to_grayscale(transforms.ToTensor()(image))
            edges = torch.abs(torch.diff(gray, dim=-1))
            texture_complexity = torch.mean(edges).item()
            
            return {
                'dimensions': (width, height),
                'aspect_ratio': aspect_ratio,
                'brightness': brightness,
                'contrast': contrast,
                'texture_complexity': texture_complexity,
                'dominant_colors': dominant_colors[:100]  # Sample for analysis
            }
        
        except Exception as e:
            raise Exception(f"Error analyzing image characteristics: {str(e)}")
    
    def compare_to_style(self, image_features: torch.Tensor, target_style: str) -> Dict:
        """Compare image features to target art style and generate feedback."""
        if target_style not in self.art_styles:
            raise ValueError(f"Unknown art style: {target_style}. Available styles: {list(self.art_styles.keys())}")
        
        style_info = self.art_styles[target_style]
        
        # This is a simplified comparison - in a real implementation, you would
        # train the model on style-specific datasets or use transfer learning
        feedback = {
            'target_style': target_style,
            'style_characteristics': style_info['characteristics'],
            'recommendations': [],
            'color_suggestions': style_info['colors'],
            'technique_suggestions': style_info['technique']
        }
        
        # Generate recommendations based on style characteristics
        if target_style == 'impressionist':
            feedback['recommendations'] = [
                "Increase brushstroke variability for a more painterly effect",
                "Soften hard edges and blend colors more",
                "Focus on capturing natural light effects",
                "Use warmer, more natural color tones",
                "Consider outdoor or natural settings for subject matter"
            ]
        elif target_style == 'cubist':
            feedback['recommendations'] = [
                "Break down forms into geometric shapes",
                "Show multiple perspectives simultaneously",
                "Reduce color palette to 3-4 main colors",
                "Emphasize angular, fragmented compositions",
                "Focus on the underlying structure of objects"
            ]
        elif target_style == 'renaissance':
            feedback['recommendations'] = [
                "Achieve more realistic proportions and anatomy",
                "Implement stronger depth of field and perspective",
                "Use chiaroscuro (light/shadow) for form definition",
                "Apply golden ratio principles to composition",
                "Focus on classical, balanced compositions"
            ]
        elif target_style == 'expressionist':
            feedback['recommendations'] = [
                "Increase color vibrancy and contrast",
                "Distort forms to express emotion",
                "Use bolder, more visible brushstrokes",
                "Emphasize emotional content over realism",
                "Consider non-naturalistic color choices"
            ]
        elif target_style == 'baroque':
            feedback['recommendations'] = [
                "Create more dramatic lighting effects (tenebrism)",
                "Add ornate details and decorative elements",
                "Use diagonal compositions for dynamic movement",
                "Implement rich, deep color palette with golds and reds",
                "Focus on theatrical, grandiose presentation"
            ]
        
        return feedback
    
    def generate_style_feedback(self, image_path: str, target_style: str) -> Dict:
        """Main function to analyze image and provide style alignment feedback."""
        print(f"Analyzing image: {image_path}")
        print(f"Target style: {target_style}")
        
        # Extract features and analyze characteristics
        features = self.extract_features(image_path)
        characteristics = self.analyze_image_characteristics(image_path)
        feedback = self.compare_to_style(features, target_style)
        
        # Add technical analysis to feedback
        feedback['technical_analysis'] = {
            'image_characteristics': characteristics,
            'feature_vector_shape': features.shape
        }
        
        return feedback

def main():
    """Main function to run the art style analyzer."""
    parser = argparse.ArgumentParser(description='Art Style Analyzer using Vision Transformer')
    parser.add_argument('image_path', type=str, help='Path to the uploaded image')
    parser.add_argument('target_style', type=str, 
                       choices=['impressionist', 'cubist', 'renaissance', 'expressionist', 'baroque'],
                       help='Target art style to align with')
    parser.add_argument('--output', type=str, default=None, help='Output JSON file for feedback')
    
    args = parser.parse_args()
    
    # Validate image path
    if not os.path.exists(args.image_path):
        print(f"Error: Image file not found: {args.image_path}")
        return
    
    try:
        # Initialize analyzer
        analyzer = ArtStyleAnalyzer()
        
        # Generate feedback
        feedback = analyzer.generate_style_feedback(args.image_path, args.target_style)
        
        # Display results
        print("\n" + "="*60)
        print("ART STYLE ANALYSIS FEEDBACK")
        print("="*60)
        print(f"Target Style: {feedback['target_style'].title()}")
        print(f"\nStyle Characteristics:")
        for char in feedback['style_characteristics']:
            print(f"  • {char.title()}")
        
        print(f"\nRecommendations:")
        for i, rec in enumerate(feedback['recommendations'], 1):
            print(f"  {i}. {rec}")
        
        print(f"\nColor Suggestions:")
        for color in feedback['color_suggestions']:
            print(f"  • {color.title()}")
        
        print(f"\nTechnique Suggestions:")
        for technique in feedback['technique_suggestions']:
            print(f"  • {technique.title()}")
        
        # Technical details
        tech = feedback['technical_analysis']['image_characteristics']
        print(f"\nTechnical Analysis:")
        print(f"Image dimensions: {tech['dimensions']}")
        print(f"Aspect ratio: {tech['aspect_ratio']:.2f}")
        print(f"Average brightness: {tech['brightness']:.1f}")
        print(f"Contrast: {tech['contrast']:.1f}")
        print(f"Texture complexity: {tech['texture_complexity']:.3f}")
        
        # Save to file if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(feedback, f, indent=2, default=str)
            print(f"\nFeedback saved to: {args.output}")
    
    except Exception as e:
        print(f"Error during analysis: {str(e)}")

if __name__ == "__main__":
    main()
