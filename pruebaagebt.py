import json
import os
import ollama
from openai import OpenAI
import ollama


OPENAI_API_KEY = "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"

class DocumentProcessor:
    def __init__(self, ollama_model="llama3.2"):
        self.ollama_model = ollama_model
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        
        # Define the categories
        self.categories = [
            "constitucion",
            "disolucion",
            "fusion",
            "cambio de accionistas",
            "cambio de sede",
            "cambio de nombre",
            "cambio de directiva",
            "incremento o disminucion de capital",
            "no aplica"
        ]
        
        # Define the fine-tuned models and prompts for each category
        self.category_config = {
            "constitucion": {
                "model": "gpt-4",
                "prompt": "Please extract all of the relevant information about this constitution as a JSON object. Include company name, date, founders, initial capital, and business purpose."
            },
            "disolucion": {
                "model": "gpt-4",
                "prompt": "Please extract all relevant information about this dissolution as a JSON object. Include company name, dissolution date, reason for dissolution, and liquidators."
            },
            "fusion": {
                "model": "gpt-4",
                "prompt": "Please extract all relevant information about this merger as a JSON object. Include companies involved, merger date, exchange ratios, and resulting entity."
            },
            "cambio de accionistas": {
                "model": "gpt-4",
                "prompt": "Please extract all relevant information about this change in shareholders as a JSON object. Include company name, date, previous shareholders, new shareholders, and shares transferred."
            },
            "cambio de sede": {
                "model": "gpt-4",
                "prompt": "Please extract all relevant information about this change of headquarters as a JSON object. Include company name, previous address, new address, and effective date."
            },
            "cambio de nombre": {
                "model": "gpt-4",
                "prompt": "Please extract all relevant information about this name change as a JSON object. Include previous name, new name, effective date, and reason for change."
            },
            "cambio de directiva": {
                "model": "gpt-4",
                "prompt": "Please extract all relevant information about this change in directors as a JSON object. Include company name, previous directors, new directors, and effective date."
            },
            "incremento o disminucion de capital": {
                "model": "gpt-4",
                "prompt": "Please extract all relevant information about this capital increase or decrease as a JSON object. Include company name, previous capital, new capital, effective date, and reason for change."
            },
            "no aplica": {
                "model": "gpt-4",
                "prompt": "Please summarize the key points of this document as a JSON object. Extract any company names, dates, and significant legal actions mentioned."
            }
        }
    
    def detect_categories(self, text):
        """
        Use Ollama to detect one or more categories in the document.
        Returns a list of detected categories.
        """
        system_prompt = f"""
        You are a document classifier that classifies legal documents into one or more of the following categories:
        {', '.join(self.categories)}
        
        Analyze the text carefully and respond with one or more category names separated by commas.
        List ALL categories that apply to this document.
        If the document doesn't clearly fit into any category, respond with "no aplica".
        """
        
        try:
            response = ollama.chat(
                model=self.ollama_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ]
            )
            
            # Extract the categories from the response
            classification_text = response["message"]["content"].strip().lower()
            
            # Split by commas and clean up whitespace
            detected_categories = [cat.strip() for cat in classification_text.split(',')]
            
            # Validate that each category is among the expected categories
            valid_categories = []
            for detected in detected_categories:
                for category in self.categories:
                    if category in detected:
                        valid_categories.append(category)
                        break
            
            # If no valid categories found
            if not valid_categories:
                print(f"Warning: Ollama returned '{classification_text}' which doesn't match any expected category.")
                return ["no aplica"]  # Default category
                
            return valid_categories
            
        except Exception as e:
            print(f"Error classifying document with Ollama: {e}")
            return ["no aplica"]  # Default to "no aplica" in case of error
    
    def process_with_openai(self, text, category):
        """
        Process the document with the appropriate OpenAI model based on its category.
        """
        # Get the specific model and prompt for this category, or use default if not found
        config = self.category_config.get(category, self.category_config["no aplica"])
        model = config["model"]
        prompt = config["prompt"]
        
        try:
            # Create the completion without the response_format parameter to avoid errors
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": prompt + " Ensure your response is valid JSON."}, 
                    {"role": "user", "content": text}
                ],
                temperature=0.3  # Lower temperature for more consistent outputs
            )
            
            content = response.choices[0].message.content
            
            # Try to parse the response as JSON to verify it's valid
            try:
                parsed_json = json.loads(content)
                return content
            except json.JSONDecodeError:
                # If not valid JSON, attempt to fix it by extracting the JSON portion
                print(f"Warning: Response for category '{category}' is not valid JSON. Attempting to extract JSON content.")
                # Look for JSON-like content between curly braces
                if '{' in content and '}' in content:
                    json_start = content.find('{')
                    json_end = content.rfind('}') + 1
                    json_content = content[json_start:json_end]
                    try:
                        # Verify the extracted content is valid JSON
                        parsed_json = json.loads(json_content)
                        return json_content
                    except json.JSONDecodeError:
                        pass
                
                # If all attempts fail, return an error message as JSON
                return json.dumps({"error": "Could not get valid JSON from model response", 
                                  "raw_response": content[:500]})
                
        except Exception as e:
            print(f"Error processing category '{category}' with OpenAI: {e}")
            return json.dumps({"error": str(e)})
    
    def process_document(self, text):
        """
        Complete document processing pipeline: detect all applicable categories
        and process with appropriate OpenAI models for each category.
        """
        # Step 1: Detect all applicable categories
        categories = self.detect_categories(text)
        print(f"Document classified as: {', '.join(categories)}")
        
        # Step 2: Process with OpenAI for each detected category
        results = {}
        for category in categories:
            print(f"Processing category: {category}")
            result = self.process_with_openai(text, category)
            
            try:
                parsed_result = json.loads(result) if isinstance(result, str) else result
                results[category] = parsed_result
            except json.JSONDecodeError:
                results[category] = {
                    "error": "Failed to parse JSON response", 
                    "raw_response": result[:500]
                }
        
        return {
            "categories": categories,
            "results": results
        }


# Example usage
if __name__ == "__main__":
    processor = DocumentProcessor()
    
    # Example document text
    sample_text = """
    ACTA DE ASAMBLEA GENERAL EXTRAORDINARIA
    
    En la ciudad de México, el día 12 de febrero de 2023, se reunieron los accionistas de
    "Tecnologías Avanzadas, S.A." para aprobar el cambio de denominación social a 
    "Innovaciones Digitales, S.A." y la modificación del domicilio social de 
    Av. Reforma 123 a Insurgentes Sur 456, así como el aumento del capital social 
    de $1,000,000.00 MXN a $2,500,000.00 MXN.
    """
    
    # Process the document
    result = processor.process_document(sample_text)
    
    # Print the results
    print(f"Detected Categories: {', '.join(result['categories'])}")
    print("\nResults by Category:")
    for category, data in result['results'].items():
        print(f"\n--- {category.upper()} ---")
        print(json.dumps(data, indent=2, ensure_ascii=False))