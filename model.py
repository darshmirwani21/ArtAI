import torch
import torchvision.transforms as transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights
from PIL import Image
import numpy as np
import argparse
import os
import json
import joblib
from typing import Dict, List, Optional

_BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
CLASSIFIER_PATH = os.path.join(_BASE_DIR, 'linear_probe.joblib')
STYLE_NAMES_PATH= os.path.join(_BASE_DIR, 'style_names_clean.joblib')
META_PATH       = os.path.join(_BASE_DIR, 'classifier_meta.json')


class ArtStyleAnalyzer:
    """
    ViT feature extractor + trained linear probe for art style classification.
    Recommendations are dynamically generated from classifier scores.
    To swap in an LLM later: replace _generate_recommendations() only.
    """

    STYLE_KB: Dict[str, Dict] = {
        'art_nouveau': {
            'characteristics': ['organic forms', 'flowing lines', 'natural elements', 'decorative patterns'],
            'colors':          ['pastels', 'soft greens', 'golden tones', 'natural palette'],
            'technique':       ['curved lines', 'botanical elements', 'ornamental details', 'sinuous forms'],
            'markers':         ['flowing curves', 'ornamental detail', 'botanical motifs', 'sinuous line work'],
            'antithesis': {
                'cubism':         "reduce geometric fragmentation and angular forms",
                'expressionism':  "tone down emotional distortion and aggressive brushwork",
                'fauvism':        "soften the wild color contrasts into a more harmonious palette",
                'futurism':       "replace dynamic speed lines with graceful, organic curves",
                'impressionism':  "add more structured decorative detail over loose atmospheric strokes",
                'neoclassicism':  "introduce organic natural motifs to break the rigid classical order",
                'surrealism':     "ground the dreamlike imagery in natural, botanical forms",
            },
            'advice': [
                "Introduce sinuous, plant-inspired curves throughout your composition",
                "Add ornamental borders or decorative frames with botanical motifs",
                "Replace hard edges with flowing, undulating lines",
                "Incorporate flowers, vines, or insects as structural design elements",
                "Use a soft pastel palette with golden or olive accents",
            ]
        },
        'cubism': {
            'characteristics': ['geometric shapes', 'multiple perspectives', 'abstract forms', 'fragmented composition'],
            'colors':          ['muted tones', 'browns', 'grays', 'limited palette'],
            'technique':       ['geometric fragmentation', 'volumetric analysis', 'simultaneous perspective', 'deconstruction'],
            'markers':         ['faceted planes', 'geometric abstraction', 'multiple viewpoints', 'muted earthy palette'],
            'antithesis': {
                'art_nouveau':    "break down the flowing organic forms into geometric planes",
                'expressionism':  "replace emotional distortion with analytical geometric structure",
                'fauvism':        "reduce the vivid color palette to muted, earthy tones",
                'futurism':       "replace movement and speed with static simultaneous perspective",
                'impressionism':  "replace soft atmospheric effects with hard geometric facets",
                'neoclassicism':  "deconstruct the classical figures into overlapping angular planes",
                'surrealism':     "replace dreamlike imagery with analytical geometric deconstruction",
            },
            'advice': [
                "Break subjects into geometric facets viewed from multiple angles simultaneously",
                "Reduce your palette to browns, grays, and ochres",
                "Flatten depth — show front, side, and top views of the same object at once",
                "Use intersecting planes and sharp angular lines",
                "Remove naturalistic shading; model form through adjacent geometric patches",
            ]
        },
        'expressionism': {
            'characteristics': ['distorted forms', 'emotional intensity', 'bold colors', 'inner feeling'],
            'colors':          ['vibrant', 'contrasting', 'non-naturalistic', 'intense hues'],
            'technique':       ['thick paint', 'bold strokes', 'exaggerated forms', 'emotional distortion'],
            'markers':         ['emotional distortion', 'intense non-naturalistic color', 'anguished forms', 'raw brushwork'],
            'antithesis': {
                'art_nouveau':    "push beyond decorative beauty into raw emotional distortion",
                'cubism':         "let emotional intensity drive form rather than analytical geometry",
                'fauvism':        "add psychological weight and distortion beyond pure color celebration",
                'futurism':       "channel energy inward as psychological tension rather than outward speed",
                'impressionism':  "replace gentle atmospheric light with raw emotional urgency",
                'neoclassicism':  "abandon idealised proportions for emotionally charged distortion",
                'surrealism':     "root the distortion in psychological anguish rather than dream logic",
            },
            'advice': [
                "Exaggerate and distort figures to amplify emotional state",
                "Apply paint thickly with aggressive, visible brushstrokes",
                "Use non-naturalistic color — make skies red, faces green if it serves the emotion",
                "Eliminate unnecessary detail; every element should carry emotional weight",
                "Increase contrast between light and dark to create psychological tension",
            ]
        },
        'fauvism': {
            'characteristics': ['bold color', 'loose brushwork', 'vivid hues', 'simplified forms'],
            'colors':          ['pure bright colors', 'non-naturalistic', 'vivid palette', 'contrasting hues'],
            'technique':       ['wild color', 'loose brushstrokes', 'pure color patches', 'color emphasis'],
            'markers':         ['pure unmixed color', 'non-naturalistic palette', 'simplified flat forms', 'joyful color energy'],
            'antithesis': {
                'art_nouveau':    "abandon decorative elegance for raw, uninhibited color expression",
                'cubism':         "replace analytical geometry with spontaneous pure color patches",
                'expressionism':  "shift from anguished distortion to joyful color liberation",
                'futurism':       "replace mechanical dynamism with pure color sensation",
                'impressionism':  "push beyond subtle color observation into bold unmixed color assertion",
                'neoclassicism':  "replace restrained earthy tones with pure, unmodulated vivid color",
                'surrealism':     "ground the image in pure visible color rather than psychological narrative",
            },
            'advice': [
                "Use color straight from the tube — avoid mixing into naturalistic shades",
                "Paint shadows in complementary colors, not darker versions of the local color",
                "Simplify forms to flat patches of pure color",
                "Let color define space and structure rather than line or shading",
                "Embrace visual clashes between adjacent complementary colors",
            ]
        },
        'futurism': {
            'characteristics': ['motion', 'speed', 'dynamism', 'mechanical forms'],
            'colors':          ['bright metallics', 'reds', 'blacks', 'industrial tones'],
            'technique':       ['dynamic lines', 'speed emphasis', 'angular forms', 'movement suggestion'],
            'markers':         ['speed lines', 'overlapping motion', 'mechanical energy', 'diagonal dynamism'],
            'antithesis': {
                'art_nouveau':    "replace organic stillness with mechanical speed and angular dynamism",
                'cubism':         "animate the static geometric planes with directional movement and speed",
                'expressionism':  "channel emotional energy outward as physical speed and mechanical force",
                'fauvism':        "add structural dynamism and directional force to the color energy",
                'impressionism':  "replace passive light observation with active mechanical motion",
                'neoclassicism':  "shatter classical stillness with overlapping motion blur and speed lines",
                'surrealism':     "replace dream stillness with aggressive mechanical dynamism",
            },
            'advice': [
                "Use repeated overlapping forms to suggest movement through time",
                "Add diagonal lines and vectors that imply speed and direction",
                "Fragment moving subjects — show multiple positions simultaneously",
                "Incorporate mechanical, industrial, or urban elements as subject matter",
                "Use high contrast between dark backgrounds and bright moving forms",
            ]
        },
        'impressionism': {
            'characteristics': ['loose brushstrokes', 'soft edges', 'natural light', 'fleeting moments'],
            'colors':          ['warm tones', 'pastels', 'natural palette', 'light colors'],
            'technique':       ['visible brush strokes', 'wet-on-wet', 'broken color', 'light capture'],
            'markers':         ['soft atmospheric light', 'broken color', 'loose visible brushwork', 'outdoor scenes'],
            'antithesis': {
                'art_nouveau':    "loosen the decorative structure into spontaneous light-catching strokes",
                'cubism':         "dissolve the geometric planes into soft atmospheric brushwork",
                'expressionism':  "soften emotional intensity into gentle observation of natural light",
                'fauvism':        "moderate the vivid color into subtle natural light harmonies",
                'futurism':       "replace mechanical speed with quiet observation of a fleeting moment",
                'neoclassicism':  "break rigid classical structure into loose, light-filled brushstrokes",
                'surrealism':     "replace dreamlike narrative with direct sensory observation of light",
            },
            'advice': [
                "Work with short, visible dabs of paint rather than smooth blended strokes",
                "Paint the light on surfaces, not the surfaces themselves",
                "Use broken color — place complementary colors side by side rather than blending",
                "Soften all hard edges; let forms dissolve into their surroundings",
                "Choose fleeting outdoor light as your subject — dawn, dusk, water reflections",
            ]
        },
        'neoclassicism': {
            'characteristics': ['classical composition', 'ordered forms', 'historical subjects', 'idealized figures'],
            'colors':          ['earthy tones', 'neutral palette', 'subdued colors', 'harmonious palette'],
            'technique':       ['linear perspective', 'balanced composition', 'classical forms', 'precise line'],
            'markers':         ['idealized figures', 'balanced symmetry', 'restrained palette', 'classical references'],
            'antithesis': {
                'art_nouveau':    "remove organic decoration and restore classical geometric order",
                'cubism':         "replace geometric abstraction with idealized classical proportion",
                'expressionism':  "replace emotional distortion with calm, rational idealized form",
                'fauvism':        "restrain the vivid palette into harmonious earthy classical tones",
                'futurism':       "replace mechanical dynamism with timeless classical stillness",
                'impressionism':  "tighten loose brushwork into precise classical line and form",
                'surrealism':     "ground dreamlike elements in rational classical compositional order",
            },
            'advice': [
                "Structure your composition with strict symmetry and balanced weight",
                "Draw figures with precise, idealised proportions referencing classical sculpture",
                "Use a restrained palette of earthy ochres, grays, and warm neutrals",
                "Employ clear linear perspective with a defined horizon and vanishing points",
                "Choose historical, mythological, or heroic subject matter",
            ]
        },
        'surrealism': {
            'characteristics': ['dreamlike imagery', 'unexpected combinations', 'subconscious elements', 'bizarre juxtaposition'],
            'colors':          ['varied palette', 'unexpected combinations', 'vibrant and muted', 'contrasting'],
            'technique':       ['detailed realism', 'impossible scenarios', 'symbolic elements', 'dreamlike quality'],
            'markers':         ['impossible juxtaposition', 'hyperreal detail in unreal context', 'dreamlike narrative', 'psychological symbolism'],
            'antithesis': {
                'art_nouveau':    "disrupt decorative harmony with irrational, dreamlike juxtapositions",
                'cubism':         "replace analytical geometry with irrational dream logic",
                'expressionism':  "replace raw emotional expression with cool detached dream narration",
                'fauvism':        "add psychological narrative and symbolic depth to the color energy",
                'futurism':       "replace mechanical speed with suspended dreamlike stillness",
                'impressionism':  "replace light observation with psychological interior imagery",
                'neoclassicism':  "subvert classical order with irrational impossible scenarios",
            },
            'advice': [
                "Combine two unrelated objects in a single scene with photographic realism",
                "Use precise, detailed rendering to make impossible scenarios feel believable",
                "Introduce unexpected scale relationships — giant objects in domestic spaces",
                "Add symbolic objects that carry psychological or subconscious meaning",
                "Treat the picture plane as a stage for the logic of dreams, not waking life",
            ]
        },
    }

    def __init__(self, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading Vision Transformer on {self.device}...")
        self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        self.vit.eval()
        self.vit.to(self.device)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.art_styles = {k: {
            'characteristics': v['characteristics'],
            'colors':          v['colors'],
            'technique':       v['technique'],
        } for k, v in self.STYLE_KB.items()}
        self.classifier      = None
        self.trained_styles: List[str] = []
        self._load_classifier()

    def _load_classifier(self):
        if os.path.exists(CLASSIFIER_PATH) and os.path.exists(STYLE_NAMES_PATH):
            self.classifier     = joblib.load(CLASSIFIER_PATH)
            raw                 = joblib.load(STYLE_NAMES_PATH)
            self.trained_styles = [s.lower() for s in raw]
            print(f"Classifier loaded — knows: {self.trained_styles}")
        else:
            print("No classifier found — style detection disabled.")

    def extract_features(self, image_input) -> torch.Tensor:
        image  = self._load_image(image_input)
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            x = self.vit._process_input(tensor)
            n = x.shape[0]
            x = torch.cat((self.vit.class_token.expand(n, -1, -1), x), dim=1)
            x = self.vit.encoder(x)
            features = torch.mean(x, dim=1)
        return features.cpu()

    def analyze_image_characteristics(self, image_input) -> Dict:
        image = self._load_image(image_input)
        w, h  = image.size
        arr   = np.array(image)
        gray  = transforms.ToTensor()(image.convert('L'))
        edges = torch.abs(torch.diff(gray, dim=-1))
        return {
            'dimensions':         (w, h),
            'aspect_ratio':       round(w / h, 4),
            'brightness':         float(np.mean(arr)),
            'contrast':           float(np.std(arr)),
            'texture_complexity': float(torch.mean(edges).item()),
            'dominant_colors':    arr.reshape(-1, 3)[:100].tolist(),
        }

    def classify(self, features: torch.Tensor) -> Dict:
        if self.classifier is None:
            return {'predicted_style': None, 'confidence': None, 'all_scores': None}
        feat_np       = features.numpy()
        predicted_idx = self.classifier.predict(feat_np)[0]
        proba         = self.classifier.predict_proba(feat_np)[0]
        predicted_style = self.trained_styles[predicted_idx]
        all_scores = {
            self.trained_styles[i]: round(float(p), 4)
            for i, p in enumerate(proba)
        }
        return {
            'predicted_style': predicted_style,
            'confidence':      round(float(proba[predicted_idx]), 4),
            'all_scores':      dict(sorted(all_scores.items(), key=lambda x: -x[1])),
        }

    # ── TO SWAP IN AN LLM: replace only this method body ─────────────────────
    def _generate_recommendations(
        self,
        target_style:    str,
        detected_style:  Optional[str],
        confidence:      Optional[float],
        all_scores:      Optional[Dict[str, float]],
        characteristics: Dict,
    ) -> List[str]:
        kb     = self.STYLE_KB
        target = target_style.lower()
        recs   = []

        if detected_style and confidence is not None and all_scores is not None:
            detected     = detected_style.lower()
            target_score = all_scores.get(target, 0.0)

            if confidence >= 0.55 and detected != target:
                pivot = kb[target]['antithesis'].get(detected, '')
                if pivot:
                    recs.append(
                        f"Your painting strongly reads as {detected.title()} "
                        f"({int(confidence*100)}% confidence) — to shift toward "
                        f"{target.title()}, {pivot}."
                    )
                else:
                    recs.append(
                        f"Your work currently aligns most closely with "
                        f"{detected.title()} — the tips below will help bridge it toward {target.title()}."
                    )
            elif target_score >= 0.5:
                recs.append(
                    f"Your painting already shows strong {target.title()} qualities "
                    f"({int(target_score*100)}%) — these refinements will push it further."
                )
            elif confidence < 0.35:
                top_two = list(all_scores.keys())[:2]
                recs.append(
                    f"Your painting has an eclectic visual signature — the classifier sees traces of "
                    f"{top_two[0].title()} and {top_two[1].title()}. "
                    f"The tips below will help you commit more clearly to {target.title()}."
                )
            elif detected == target:
                recs.append(
                    f"Great foundation — the AI already detects {target.title()} "
                    f"({int(confidence*100)}%). Focus on these refinements:"
                )

        recs.extend(kb[target]['advice'])

        # Image characteristic nudges
        brightness = characteristics.get('brightness', 128)
        contrast   = characteristics.get('contrast', 50)
        texture    = characteristics.get('texture_complexity', 0.05)

        bright_tips = {
            'art_nouveau':   ("dark",   "Lighten your palette — Art Nouveau favours soft, luminous tones."),
            'impressionism': ("dark",   "Impressionism thrives on light — brighten your overall palette."),
            'fauvism':       ("dark",   "Fauvism celebrates bright vivid color — increase your overall luminosity."),
            'expressionism': ("bright", "Consider darker, more brooding tones to amplify emotional weight."),
            'surrealism':    ("bright", "A slightly darker, more shadowy palette can heighten dreamlike unease."),
        }
        contrast_tips = {
            'expressionism': ("low",  "Increase contrast sharply — expressionism depends on tonal drama."),
            'futurism':      ("low",  "Stronger contrast will sharpen the sense of speed and energy."),
            'neoclassicism': ("high", "Soften contrast slightly for a more harmonious classical balance."),
            'impressionism': ("high", "Reduce contrast — impressionism blends light and shadow softly."),
        }
        texture_tips = {
            'impressionism': ("smooth", "Add more visible brushwork — impressionism is built on textured paint surface."),
            'expressionism': ("smooth", "Increase impasto and visible stroke energy to match expressionist physicality."),
            'neoclassicism': ("rough",  "Smooth out your surface — neoclassicism favours refined, precise execution."),
            'art_nouveau':   ("rough",  "Refine your line quality — Art Nouveau demands smooth, controlled curves."),
        }

        if target in bright_tips:
            cond, tip = bright_tips[target]
            if cond == "dark"   and brightness < 80:  recs.append(tip)
            if cond == "bright" and brightness > 190: recs.append(tip)
        if target in contrast_tips:
            cond, tip = contrast_tips[target]
            if cond == "low"  and contrast < 35: recs.append(tip)
            if cond == "high" and contrast > 90: recs.append(tip)
        if target in texture_tips:
            cond, tip = texture_tips[target]
            if cond == "smooth" and texture < 0.03: recs.append(tip)
            if cond == "rough"  and texture > 0.12: recs.append(tip)

        return recs

    def compare_to_style(self, target_style: str, classification: Dict, characteristics: Dict) -> Dict:
        key = target_style.lower()
        if key not in self.STYLE_KB:
            raise ValueError(f"Unknown style '{target_style}'. Available: {list(self.STYLE_KB.keys())}")
        info = self.STYLE_KB[key]
        recs = self._generate_recommendations(
            target_style    = key,
            detected_style  = classification.get('predicted_style'),
            confidence      = classification.get('confidence'),
            all_scores      = classification.get('all_scores'),
            characteristics = characteristics,
        )
        return {
            'target_style':          key,
            'style_characteristics': info['characteristics'],
            'recommendations':       recs,
            'color_suggestions':     info['colors'],
            'technique_suggestions': info['technique'],
        }

    def generate_style_feedback(self, image_input, target_style: str) -> Dict:
        features        = self.extract_features(image_input)
        characteristics = self.analyze_image_characteristics(image_input)
        classification  = self.classify(features)
        feedback        = self.compare_to_style(target_style, classification, characteristics)
        feedback['classification'] = classification
        feedback['technical_analysis'] = {
            'image_characteristics': characteristics,
            'feature_vector_shape':  list(features.shape),
        }
        return feedback

    @staticmethod
    def _load_image(image_input) -> Image.Image:
        if isinstance(image_input, Image.Image):
            return image_input.convert('RGB')
        if hasattr(image_input, 'seek'):
            image_input.seek(0)
        return Image.open(image_input).convert('RGB')


def main():
    parser = argparse.ArgumentParser(description='STYLO — Art Style Analyzer')
    parser.add_argument('image_path',   type=str)
    parser.add_argument('target_style', type=str, choices=list(ArtStyleAnalyzer.STYLE_KB.keys()))
    parser.add_argument('--output',     type=str, default=None)
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(f"Error: file not found: {args.image_path}"); return

    analyzer = ArtStyleAnalyzer()
    feedback = analyzer.generate_style_feedback(args.image_path, args.target_style)

    print("\n" + "="*60)
    clf = feedback['classification']
    if clf['predicted_style']:
        print(f"Detected : {clf['predicted_style'].title()} ({clf['confidence']:.1%})")
        for style, score in clf['all_scores'].items():
            bar = '█' * int(score * 20)
            print(f"  {style:<18} {score:.1%}  {bar}")

    print(f"\nTarget   : {feedback['target_style'].title()}")
    print("\nRecommendations:")
    for i, r in enumerate(feedback['recommendations'], 1):
        print(f"  {i}. {r}")

    tech = feedback['technical_analysis']['image_characteristics']
    print(f"\nImage    : {tech['dimensions'][0]}x{tech['dimensions'][1]}px  "
          f"brightness={tech['brightness']:.1f}  contrast={tech['contrast']:.1f}")

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(feedback, f, indent=2, default=str)
        print(f"\nSaved to: {args.output}")


if __name__ == '__main__':
    main()