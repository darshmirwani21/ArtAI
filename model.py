
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
            'art_nouveau': {
                'characteristics': ['organic forms', 'flowing lines', 'natural elements', 'decorative patterns'],
                'colors': ['pastels', 'soft greens', 'golden tones', 'natural palette'],
                'technique': ['curved lines', 'botanical elements', 'ornamental details', 'sinuous forms']
            },
            'cubism': {
                'characteristics': ['geometric shapes', 'multiple perspectives', 'abstract forms', 'fragmented composition'],
                'colors': ['muted tones', 'browns', 'grays', 'limited palette'],
                'technique': ['geometric fragmentation', 'volumetric analysis', 'simultaneous perspective', 'deconstruction']
            },
            'expressionism': {
                'characteristics': ['distorted forms', 'emotional intensity', 'bold colors', 'inner feeling'],
                'colors': ['vibrant', 'contrasting', 'non-naturalistic', 'intense hues'],
                'technique': ['thick paint', 'bold strokes', 'exaggerated forms', 'emotional distortion']
            },
            'fauvism': {
                'characteristics': ['bold color', 'loose brushwork', 'vivid hues', 'simplified forms'],
                'colors': ['pure bright colors', 'non-naturalistic', 'vivid palette', 'contrasting hues'],
                'technique': ['wild color', 'loose brushstrokes', 'pure color patches', 'color emphasis']
            },
            'futurism': {
                'characteristics': ['motion', 'speed', 'dynamism', 'mechanical forms', 'violence'],
                'colors': ['bright metallics', 'reds', 'blacks', 'industrial tones'],
                'technique': ['dynamic lines', 'speed emphasis', 'angular forms', 'movement suggestion']
            },
            'impressionism': {
                'characteristics': ['loose brushstrokes', 'soft edges', 'natural light', 'fleeting moments'],
                'colors': ['warm tones', 'pastels', 'natural palette', 'light colors'],
                'technique': ['visible brush strokes', 'wet-on-wet', 'broken color', 'light capture']
            },
            'neoclassicism': {
                'characteristics': ['classical composition', 'ordered forms', 'historical subjects', 'idealized figures'],
                'colors': ['earthy tones', 'neutral palette', 'subdued colors', 'harmonious palette'],
                'technique': ['linear perspective', 'balanced composition', 'classical forms', 'precise line']
            },
            'surrealism': {
                'characteristics': ['dreamlike imagery', 'unexpected combinations', 'subconscious elements', 'bizarre juxtaposition'],
                'colors': ['varied palette', 'unexpected combinations', 'vibrant and muted', 'contrasting'],
                'technique': ['detailed realism', 'impossible scenarios', 'symbolic elements', 'dreamlike quality']
            }
        }
    
    def extract_features(self, image_input) -> torch.Tensor:
        """Extract visual features from an uploaded image using the Vision Transformer.
        Accepts either a file path (str) or a file-like object (BytesIO, file object).
        """
        try:
            # Load and preprocess image. Supports:
            # - PIL.Image.Image instances (already loaded Pillow images)
            # - file paths (str)
            # - file-like objects (BytesIO, Werkzeug FileStorage stream, etc.)
            if isinstance(image_input, Image.Image):
                image = image_input.convert('RGB')
            else:
                # Reset stream position if it's a file-like object
                if hasattr(image_input, 'seek'):
                    image_input.seek(0)
                image = Image.open(image_input).convert('RGB')
            
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Extract features from the Vision Transformer
            with torch.no_grad():
                # For torchvision ViT, we need to extract features before the classification head
                # The model outputs (logits, features) when we access intermediate layers
                x = input_tensor
                # Reshape and permute as the ViT expects
                n, c, h, w = x.shape
                p = self.model.image_size // self.model.patch_size
                x = self.model._process_input(x)
                n, _, c = x.shape
                
                # Add class token
                batch_class_token = self.model.class_token.expand(n, -1, -1)
                x = torch.cat((batch_class_token, x), dim=1)
                
                # Apply transformer blocks
                x = self.model.encoder(x)
                
                # Global average pooling on all tokens (including class token)
                features = torch.mean(x, dim=1)
            
            return features.cpu()
        
        except Exception as e:
            raise Exception(f"Error processing image: {str(e)}")
    
    def analyze_image_characteristics(self, image_input) -> Dict:
        """Analyze visual characteristics of the uploaded image.
        Accepts either a file path (str) or a file-like object (BytesIO, file object).
        """
        try:
            # Load image. Support PIL.Image.Image, file path, or file-like objects.
            if isinstance(image_input, Image.Image):
                image = image_input.convert('RGB')
            else:
                # Reset stream position if it's a file-like object
                if hasattr(image_input, 'seek'):
                    image_input.seek(0)
                image = Image.open(image_input).convert('RGB')
            
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
            # Convert PIL Image to grayscale, then to tensor
            gray_image = image.convert('L')  # Convert to grayscale PIL Image
            gray = transforms.ToTensor()(gray_image)
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
        
        # This is a simplified comparison - in a real implementation, you would train model on style-specific datasets
        # Here, we provide generic feedback based on style characteristics
        feedback = {
            'target_style': target_style,
            'style_characteristics': style_info['characteristics'],
            'recommendations': [],
            'color_suggestions': style_info['colors'],
            'technique_suggestions': style_info['technique']
        }
        
        # Generate recommendations based on style characteristics
        if target_style == 'art_nouveau':
            feedback['recommendations'] = [
                "Incorporate organic, flowing curves into your composition",
                "Use natural elements like plants, flowers, or insects as motifs",
                "Apply decorative patterns and ornamental details throughout",
                "Employ soft, pastel colors with occasional golden accents",
                "Focus on sinuous, undulating lines and graceful forms"
            ]
        elif target_style == 'cubism':
            feedback['recommendations'] = [
                "Break down forms into geometric shapes (cubes, spheres, cones)",
                "Show multiple perspectives of the same object simultaneously",
                "Reduce color palette to 3-4 main colors",
                "Emphasize angular, fragmented compositions",
                "Focus on the underlying structure and volume of objects"
            ]
        elif target_style == 'expressionism':
            feedback['recommendations'] = [
                "Increase color vibrancy and emotional intensity",
                "Distort forms to express inner emotion rather than reality",
                "Use bold, visible brushstrokes with thick paint application",
                "Emphasize emotional content over accurate representation",
                "Incorporate non-naturalistic color choices for emotional effect"
            ]
        elif target_style == 'fauvism':
            feedback['recommendations'] = [
                "Use pure, bright colors directly from the palette",
                "Apply bold, non-naturalistic color choices freely",
                "Simplify forms while maintaining recognizability",
                "Apply loose, expressive brushwork",
                "Emphasize color over accurate rendering of reality"
            ]
        elif target_style == 'futurism':
            feedback['recommendations'] = [
                "Emphasize movement, speed, and dynamism in your composition",
                "Incorporate mechanical and industrial elements",
                "Use diagonal and angular lines to suggest motion",
                "Apply bright, metallic, and contrasting colors",
                "Focus on capturing energy and forward momentum"
            ]
        elif target_style == 'impressionism':
            feedback['recommendations'] = [
                "Increase brushstroke variability for a more painterly effect",
                "Soften hard edges and blend colors more smoothly",
                "Focus on capturing natural light effects and atmosphere",
                "Use warmer, more natural color tones",
                "Consider outdoor settings or fleeting moments as subjects"
            ]
        elif target_style == 'neoclassicism':
            feedback['recommendations'] = [
                "Achieve classical, idealized proportions and forms",
                "Implement balanced, symmetrical composition",
                "Use precise lines and careful geometric arrangement",
                "Apply subdued, earthy color tones harmoniously",
                "Focus on historical or mythological subjects with noble themes"
            ]
        elif target_style == 'surrealism':
            feedback['recommendations'] = [
                "Combine unexpected or incongruous elements in your composition",
                "Create dreamlike, illogical scenarios that provoke thought",
                "Use detailed, realistic rendering for impossible situations",
                "Incorporate symbolic and psychological elements",
                "Blend conscious and subconscious imagery for surreal effects"
            ]
        
        return feedback
    
    def generate_style_feedback(self, image_input, target_style: str) -> Dict:
        """Main function to analyze image and provide style alignment feedback.
        Accepts either a file path (str) or a file-like object (BytesIO, file object).
        """
        # Extract features and analyze characteristics
        features = self.extract_features(image_input)
        characteristics = self.analyze_image_characteristics(image_input)
        feedback = self.compare_to_style(features, target_style)
        
        # Add technical analysis to feedback
        feedback['technical_analysis'] = {
            'image_characteristics': characteristics,
            'feature_vector_shape': list(features.shape)
        }
        
        return feedback

class ArtStyleTrainer:
    """
    Trainer class for fine-tuning the Vision Transformer on art style images
    from Google Drive or local folders.
    """
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.art_styles = ['art_nouveau', 'cubism', 'expressionism', 'fauvism', 
                          'futurism', 'impressionism', 'neoclassicism', 'surrealism']
    
    def load_images_from_folder(self, base_path: str, img_size: int = 224) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load images from organized folder structure.
        Expected structure: base_path/style_name/*.jpg
        """
        X = []
        y = []
        style_counts = {style: 0 for style in self.art_styles}
        
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        for style_idx, style in enumerate(self.art_styles):
            style_path = os.path.join(base_path, style)
            if not os.path.exists(style_path):
                print(f"Warning: {style_path} not found, skipping this style.")
                continue
            
            for img_file in os.listdir(style_path):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')):
                    try:
                        img_path = os.path.join(style_path, img_file)
                        img = Image.open(img_path).convert('RGB')
                        img_tensor = transform(img)
                        X.append(img_tensor)
                        y.append(style_idx)
                        style_counts[style] += 1
                    except Exception as e:
                        print(f"Error loading {img_path}: {e}")
        
        print("\nImages loaded per style:")
        for style, count in style_counts.items():
            print(f"  {style}: {count}")
        
        return X, np.array(y), self.art_styles
    
    def train_on_drive_images(self, drive_path: str = 'art_styles_data', epochs: int = 10):
        """
        Train the model on images from Google Drive or local folder.
        For Google Colab, mount drive first:
        from google.colab import drive
        drive.mount('/content/drive')
        Then pass the path to your art styles folder.
        """
        print("Loading images from folder...")
        X, y, style_names = self.load_images_from_folder(drive_path)
        
        if len(X) == 0:
            print("No images found. Please check your folder structure.")
            return
        
        print(f"Loaded {len(X)} images across {len(self.art_styles)} styles.")
        print("Training integration ready - use the EDA notebook for full training pipeline.")

def main():
    """Main function to run the art style analyzer."""
    parser = argparse.ArgumentParser(description='Art Style Analyzer using Vision Transformer')
    parser.add_argument('image_path', type=str, help='Path to the uploaded image')
    parser.add_argument('target_style', type=str, 
                       choices=['art_nouveau', 'cubism', 'expressionism', 'fauvism', 
                               'futurism', 'impressionism', 'neoclassicism', 'surrealism'],
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

