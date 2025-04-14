import os
from googletrans import Translator

def translate_text(text, dest_language='ms'):
    """
    Translate text to the specified language
    
    Args:
        text (str): Text to translate
        dest_language (str): Destination language code (default: 'ms' for Malay)
        
    Returns:
        str: Translated text
    """
    translator = Translator()
    try:
        result = translator.translate(text, dest=dest_language)
        return result.text
    except Exception as e:
        print(f"Translation error: {e}")
        return text  # Return original text if translation fails

def translate_file(input_file, output_file, dest_language='ms'):
    """
    Translate content from input file to output file
    
    Args:
        input_file (str): Path to input file
        output_file (str): Path to output file
        dest_language (str): Destination language code (default: 'ms' for Malay)
    """
    if not os.path.exists(input_file):
        print(f"Input file {input_file} does not exist")
        return
    
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split content into manageable chunks to avoid translation limits
    # Translate by paragraphs to maintain context
    paragraphs = content.split('\n\n')
    translated_paragraphs = []
    
    for i, paragraph in enumerate(paragraphs):
        if paragraph.strip():
            # Skip translation for headers (lines starting with #)
            if paragraph.strip().startswith('#'):
                translated_paragraphs.append(paragraph)
            else:
                translated = translate_text(paragraph, dest_language)
                translated_paragraphs.append(translated)
            
            # Print progress
            if (i + 1) % 5 == 0:
                print(f"Translated {i + 1}/{len(paragraphs)} paragraphs")
    
    translated_content = '\n\n'.join(translated_paragraphs)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(translated_content)
    
    print(f"Translation completed. Output saved to {output_file}")

if __name__ == "__main__":
    input_file = "/home/ubuntu/organized_content.md"
    output_file = "/home/ubuntu/translated_content.md"
    
    translate_file(input_file, output_file, 'ms')
