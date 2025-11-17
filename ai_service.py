"""
AI Service for image analysis using LM Studio
Handles communication with local LM Studio API
"""

import requests
import base64
import json
from typing import Dict, Tuple, Optional
from pathlib import Path


class AIService:
    def __init__(self, lm_studio_url: str = "http://localhost:1234"):
        self.lm_studio_url = lm_studio_url
        self.api_endpoint = f"{lm_studio_url}/v1/chat/completions"

        # Different description styles/prompts
        self.prompts = {
            'classic': {
                'name': 'Classic',
                'description': 'Concise and factual (1-3 sentences)',
                'prompt': """Analyze this image and provide:
1. A concise description (1-3 sentences) describing what you see
2. 5-10 relevant tags (keywords) as a list
3. A suggested filename (descriptive, lowercase, use underscores, no spaces, max 50 chars, WITHOUT file extension)

You must respond with ONLY a valid JSON object in this exact format:
{
  "description": "your description here",
  "tags": ["tag1", "tag2", "tag3"],
  "suggested_filename": "descriptive_filename_here"
}

CRITICAL INSTRUCTIONS:
- Your ENTIRE response must be ONLY the JSON object above
- Do NOT add any explanations before or after the JSON
- Do NOT use markdown code blocks (no ```json```)
- Do NOT add any commentary or additional text
- Just the raw JSON object and nothing else

Guidelines:
- Keep description brief and factual
- Keep tags lowercase, prefer single words, ensure they are unique and relevant
- Filename should be descriptive but concise
- Use underscores instead of spaces in filename
- Do NOT include file extension in suggested_filename
- Ensure the JSON is valid with no trailing commas or syntax errors"""
            },

            'artistic': {
                'name': 'Artistic',
                'description': 'Detailed, poetic, and creative description',
                'prompt': """Analyze this image with artistic detail and provide:
1. A detailed, artistic description (4-6 paragraphs) that captures the mood, atmosphere, colors, composition, and emotional impact. Be poetic and evocative.
2. 8-15 relevant tags including mood, style, and technical aspects
3. A suggested filename (descriptive, lowercase, use underscores, no spaces, max 50 chars, WITHOUT file extension)

You must respond with ONLY a valid JSON object in this exact format:
{
  "description": "your detailed artistic description here",
  "tags": ["tag1", "tag2", "tag3"],
  "suggested_filename": "descriptive_filename_here"
}

CRITICAL INSTRUCTIONS:
- Your ENTIRE response must be ONLY the JSON object above
- Do NOT add any explanations before or after the JSON
- Do NOT use markdown code blocks (no ```json```)
- Do NOT add any commentary or additional text
- Just the raw JSON object and nothing else

Guidelines:
- Make the description rich, detailed, and atmospheric
- Include details about lighting, composition, mood, colors
- Use vivid, descriptive language
- Tags should include artistic and technical terms; keep them lowercase, prefer single words, ensure they are unique and relevant
- Filename should be descriptive but concise
- Use underscores instead of spaces in filename
- Do NOT include file extension in suggested_filename
- Ensure the JSON is valid with no trailing commas or syntax errors"""
            },

            'spicy': {
                'name': 'Spicy',
                'description': 'Provocative and attention-grabbing style',
                'prompt': """Analyze this image with a bold, provocative style and provide:
1. A captivating description (3-5 paragraphs) that's attention-grabbing, bold, and engaging. Emphasize the most striking and alluring aspects.
2. 8-15 tags including mood and aesthetic qualities
3. A suggested filename (descriptive, evocative, lowercase, use underscores, no spaces, max 50 chars, WITHOUT file extension)

You must respond with ONLY a valid JSON object in this exact format:
{
  "description": "your bold description here",
  "tags": ["tag1", "tag2", "tag3"],
  "suggested_filename": "descriptive_filename_here"
}

CRITICAL INSTRUCTIONS:
- Your ENTIRE response must be ONLY the JSON object above
- Do NOT add any explanations before or after the JSON
- Do NOT use markdown code blocks (no ```json```)
- Do NOT add any commentary or additional text
- Just the raw JSON object and nothing else

Guidelines:
- Make the description bold, captivating, and more provocative with sensual, tantalizing language to heighten allure and intensity
- Emphasize visual appeal and striking elements
- Use confident, engaging language
- Focus on what makes the image stand out
- Tags should include mood and aesthetics; keep them lowercase, prefer single words, ensure they are unique and relevant
- Filename should be descriptive but concise
- Use underscores instead of spaces in filename
- Do NOT include file extension in suggested_filename
- Ensure the JSON is valid with no trailing commas or syntax errors"""
            },

            'social': {
                'name': 'Social Media',
                'description': 'Optimized for Instagram, Facebook, Twitter',
                'prompt': """Analyze this image for social media posting and provide:
1. A social media-ready description (2-4 paragraphs) that's engaging, relatable, and perfect for Instagram, Facebook, or Twitter. Use conversational tone and be authentic.
2. 10-15 trending hashtags and relevant keywords (include the # for hashtags)
3. A suggested filename (catchy, descriptive, lowercase, use underscores, no spaces, max 50 chars, WITHOUT file extension)

You must respond with ONLY a valid JSON object in this exact format:
{
  "description": "your social media description here",
  "tags": ["#hashtag1", "#hashtag2", "keyword1"],
  "suggested_filename": "descriptive_filename_here"
}

CRITICAL INSTRUCTIONS:
- Your ENTIRE response must be ONLY the JSON object above
- Do NOT add any explanations before or after the JSON
- Do NOT use markdown code blocks (no ```json```)
- Do NOT add any commentary or additional text
- Just the raw JSON object and nothing else

Guidelines:
- Write in a friendly, conversational tone
- Make it shareable and relatable
- Tags should include the # for hashtags where appropriate; keep them lowercase, prefer single words or phrases, ensure they are unique and relevant
- Consider what would perform well on social platforms
- Filename should be descriptive but concise
- Use underscores instead of spaces in filename
- Do NOT include file extension in suggested_filename
- Ensure the JSON is valid with no trailing commas or syntax errors"""
            },

            'tags': {
                'name': 'Tags Only',
                'description': 'Generate only tags/keywords without description',
                'prompt': """Analyze this image and provide ONLY tags/keywords.

You must respond with ONLY a valid JSON object in this exact format:
{
  "description": "",
  "tags": ["tag1", "tag2", "tag3", "tag4", "tag5", "tag6", "tag7", "tag8", "tag9", "tag10"],
  "suggested_filename": ""
}

CRITICAL INSTRUCTIONS:
- Your ENTIRE response must be ONLY the JSON object above
- Generate 8-15 relevant, descriptive tags/keywords for this image
- Leave description empty (empty string "")
- Leave suggested_filename empty (empty string "")
- Do NOT add any explanations before or after the JSON
- Do NOT use markdown code blocks (no ```json```)
- Do NOT add any commentary or additional text
- Just the raw JSON object and nothing else

Guidelines for tags:
- Keep tags lowercase
- Prefer single words (use compound words if needed like "golden_hour")
- Include objects, colors, mood, setting, composition style
- Ensure tags are unique and highly relevant
- No hashtags (#) - just plain keywords
- Ensure the JSON is valid with no trailing commas or syntax errors"""
            },

            'custom': {
                'name': 'Custom',
                'description': 'Use your own custom prompt',
                'prompt': None  # Will be provided by user
            }
        }

    def get_available_styles(self) -> Dict[str, Dict]:
        """Get all available description styles"""
        return {
            key: {
                'name': value['name'],
                'description': value['description']
            }
            for key, value in self.prompts.items()
        }

    def check_connection(self) -> Tuple[bool, str]:
        """Check if LM Studio is running and accessible"""
        try:
            response = requests.get(f"{self.lm_studio_url}/v1/models", timeout=5)
            if response.status_code == 200:
                return True, "LM Studio is connected"
            else:
                return False, f"LM Studio returned status {response.status_code}"
        except requests.exceptions.ConnectionError:
            return False, "Cannot connect to LM Studio. Is it running?"
        except requests.exceptions.Timeout:
            return False, "Connection to LM Studio timed out"
        except Exception as e:
            return False, f"Error: {str(e)}"
    
    def analyze_image(self, image_path: str, style: str = 'classic', custom_prompt: str = None) -> Optional[Dict]:
        """
        Analyze image and return description and tags

        Args:
            image_path: Path to the image file
            style: Description style ('classic', 'artistic', 'spicy', 'social', 'custom')
            custom_prompt: Custom prompt text (only used if style='custom')

        Returns: {'description': str, 'tags': List[str], 'suggested_filename': str} or None on error
        """
        try:
            # Read and encode image
            with open(image_path, 'rb') as f:
                image_data = f.read()

            base64_image = base64.b64encode(image_data).decode('utf-8')

            # Determine image format
            ext = Path(image_path).suffix.lower()
            mime_type = {
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.png': 'image/png',
                '.gif': 'image/gif',
                '.webp': 'image/webp',
                '.bmp': 'image/bmp'
            }.get(ext, 'image/jpeg')

            # Get prompt based on style
            if style == 'custom' and custom_prompt:
                # Wrap custom prompt with JSON instructions
                prompt = f"""{custom_prompt}

You must respond with ONLY a valid JSON object in this exact format:
{{
  "description": "your description here",
  "tags": ["tag1", "tag2", "tag3"],
  "suggested_filename": "descriptive_filename_here"
}}

CRITICAL INSTRUCTIONS:
- Your ENTIRE response must be ONLY the JSON object above
- Do NOT add any explanations before or after the JSON
- Do NOT use markdown code blocks (no ```json```)
- Do NOT add any commentary or additional text
- Just the raw JSON object and nothing else
- Ensure the JSON is valid with no trailing commas or syntax errors"""
            elif style in self.prompts:
                prompt = self.prompts[style]['prompt']
            else:
                # Fallback to classic
                prompt = self.prompts['classic']['prompt']

            print(f"Using '{style}' style for analysis")

            # Prepare API request
            payload = {
                "model": "llava",  # or whatever vision model is loaded
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 500,
                "temperature": 0.7
            }
            
            print(f"Sending analysis request to {self.api_endpoint}")
            
            # Send request to LM Studio
            response = requests.post(
                self.api_endpoint,
                json=payload,
                timeout=120  # 2 minutes timeout for slow models
            )
            
            print(f"Response status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                
                # Extract content from response
                if 'choices' not in result or len(result['choices']) == 0:
                    print(f"Invalid response structure: {result}")
                    return None
                
                content = result['choices'][0]['message']['content']
                print(f"AI response: {content[:200]}...")
                
                # Try to parse JSON from content
                parsed = self._extract_json(content)
                
                if parsed:
                    result = {
                        'description': parsed.get('description', ''),
                        'tags': parsed.get('tags', []),
                        'suggested_filename': parsed.get('suggested_filename', '')
                    }
                    print(f"AI suggested filename: {result.get('suggested_filename', 'none')}")
                    return result
                else:
                    # Fallback: treat whole response as description
                    print("Warning: Could not parse JSON, using raw response")
                    return {
                        'description': content.strip(),
                        'tags': [],
                        'suggested_filename': ''
                    }
            else:
                print(f"LM Studio error: {response.status_code} - {response.text}")
                return None
                
        except FileNotFoundError:
            print(f"Image file not found: {image_path}")
            return None
        except requests.exceptions.ConnectionError as e:
            print(f"Connection error: {str(e)}")
            print("Make sure LM Studio is running with local server enabled")
            return None
        except requests.exceptions.Timeout:
            print(f"Analysis timed out for {image_path}")
            return None
        except Exception as e:
            print(f"Error analyzing image: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def _extract_json(self, text: str) -> Optional[Dict]:
        """
        Extract JSON from text that might contain markdown code blocks or extra text.
        Handles cases where AI returns JSON followed by additional explanation.
        """
        import re
        
        # Try direct parse first (if text is pure JSON)
        text = text.strip()
        try:
            return json.loads(text)
        except:
            pass
        
        # Look for ```json ... ``` markdown code blocks
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except:
                pass
        
        # Look for properly nested JSON {...} block
        # This regex handles nested objects correctly
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except:
                pass
        
        # Last resort: Manual brace counting to extract complete JSON
        # This handles deeply nested structures
        try:
            start_idx = text.find('{')
            if start_idx == -1:
                return None
            
            brace_count = 0
            end_idx = start_idx
            in_string = False
            escape_next = False
            
            for i in range(start_idx, len(text)):
                char = text[i]
                
                # Handle string escaping
                if escape_next:
                    escape_next = False
                    continue
                
                if char == '\\':
                    escape_next = True
                    continue
                
                # Track if we're inside a string
                if char == '"':
                    in_string = not in_string
                    continue
                
                # Only count braces outside of strings
                if not in_string:
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_idx = i + 1
                            break
            
            if end_idx > start_idx:
                json_str = text[start_idx:end_idx]
                parsed = json.loads(json_str)
                print(f"Successfully extracted JSON using brace counting")
                return parsed
        except Exception as e:
            print(f"Brace counting extraction failed: {str(e)}")
            pass
        
        print(f"Warning: Could not extract valid JSON from response")
        return None
    
    def batch_analyze(self, image_paths: list, progress_callback=None) -> Dict[str, Dict]:
        """
        Analyze multiple images
        Returns: {image_path: {'description': str, 'tags': list}, ...}
        """
        results = {}
        total = len(image_paths)
        
        for i, path in enumerate(image_paths):
            if progress_callback:
                progress_callback(i + 1, total, path)
            
            result = self.analyze_image(path)
            results[path] = result
        
        return results
