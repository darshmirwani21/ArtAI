#!/usr/bin/env python3
"""
Example usage of the Art Style Analyzer system.
This script demonstrates how to use the ArtStyleAnalyzer class programmatically.
"""

import os
from model import ArtStyleAnalyzer

def main():
    """Example usage of the ArtStyleAnalyzer."""
    
    # Defining Image Path
    image_path = "impresion.jpg"  
    
    # Check if example image exists
    if not os.path.exists(image_path):
        print(f"Please provide an image at: {image_path}")
        print("You can use any JPEG or PNG image file.")
        return
    
    # Initialize the analyzer
    print("Initializing Art Style Analyzer...")
    analyzer = ArtStyleAnalyzer()
    
    # Available art styles
    available_styles = ['impressionist', 'cubist', 'renaissance', 'expressionist', 'baroque']
    
    print(f"\nAnalyzing image: {image_path}")
    print("Available target styles:", ", ".join(available_styles))
    
    # Ask user for target style
    target_style = input('What is your target style? \n')  
    
    try:
        print(f"\nAnalyzing for {target_style} style...")
        feedback = analyzer.generate_style_feedback(image_path, target_style)
        
        # Display results
        print("\n" + "="*50)
        print("FEEDBACK RESULTS")
        print("="*50)
        
        print(f"Target Style: {feedback['target_style'].title()}")
        
        print("\nRecommendations:")
        for i, rec in enumerate(feedback['recommendations'], 1):
            print(f"  {i}. {rec}")
        
        print("\nColor Suggestions:")
        for color in feedback['color_suggestions']:
            print(f"  • {color}")
        
        print("\nTechnique Suggestions:")
        for technique in feedback['technique_suggestions']:
            print(f"  • {technique}")
            
    except Exception as e:
        print(f"Error during analysis: {e}")

if __name__ == "__main__":
    main()
