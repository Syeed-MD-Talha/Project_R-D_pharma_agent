import asyncio
import nest_asyncio
import re
from collections import defaultdict
from google import genai
from google.genai.types import Part, Tool, GenerateContentConfig, GoogleSearch
from google.colab import files

# Apply nest_asyncio for Google Colab
nest_asyncio.apply()

# Initialize the client
API_KEY = ""
client = genai.Client(api_key=API_KEY)
model_id = "gemini-2.0-flash"

# Configure Google Search Tool
google_search_tool = Tool(
    google_search=GoogleSearch()
)

# Upload the image
print("Please upload the prescription image:")
uploaded = files.upload()
file_name = list(uploaded.keys())[0]
file_content = uploaded[file_name]

image = Part.from_bytes(
    data=file_content,
    mime_type="image/jpeg"
)

initial_prompt = """
This image contains a handwritten prescription. Please analyze it and provide:

TASK:
    Carefully analyze the uploaded prescription image and extract:
    1. Your interpretation of each medicine name in the prescription
    2. For each medicine name, assign a confidence percentage (0-100%)
    3. Format each line as "Medicine Name: [confidence]%"
    4. All medicine names (which may be handwritten in English or Bangla)
    5. Dosage instructions for each medicine (e.g., 0+0+1/2, 1+1+1, etc.)
    6. Any special instructions for taking the medicines
    
    IMPORTANT NOTES:
    - DO NOT try to validate or correct medicine names yet
    - Extract exactly what you see, even if the spelling seems unusual
    - For unclear text, provide your best interpretation and note uncertainty
    - Dosage notation typically follows patterns like 0+1+0 (morning+noon+night)
    - Write everything in english
    
    OUTPUT FORMAT:
    ```
    EXTRACTED MEDICINES:
    
    Medicine 1:
    - Name: [medicine name] ([confidence]%)
    - Dosage: [frequency pattern]
    - Instructions: [any special directions]
    
    Medicine 2:
    - Name: [medicine name] ([confidence]%)
    - Dosage: [frequency pattern]
    - Instructions: [any special directions]
    
    [continue for each medicine identified]
    ```
"""

# Async Function to generate interpretation
async def generate_interpretation(temperature):
    response = await client.aio.models.generate_content(
        model=model_id,
        contents=[initial_prompt, image],
        config=GenerateContentConfig(
            temperature=0.2
            # temperature=temperature,
        ),
    )
    return response.text

# Run multiple interpretations concurrently
async def run_parallel_interpretations(num_passes=5):
    temperatures = [0.7 + (i * 0.2) for i in range(num_passes)]
    tasks = [generate_interpretation(temp) for temp in temperatures]
    results = await asyncio.gather(*tasks)
    return results

# Generate final consolidated output
async def generate_final_output(all_interpretations):
    consolidation_prompt = """
    I have multiple interpretations of a handwritten prescription. Each interpretation attempts to extract 
    medicine names, dosages, and instructions. Please consolidate these interpretations into a single, 
    accurate list of medicines with their dosages and instructions.
    
    Here are the multiple interpretations:
    
    {INTERPRETATIONS}
    
    Based on these interpretations, please provide a final consolidated output in this exact format:
    
    Medicine 1:
    - Name: [exactly as written (don't include the power just medicine name only)]
    - Dosage: [frequency pattern]
    - Instructions: [any special directions]
    
    Medicine 2:
    - Name: [exactly as written]
    - Dosage: [frequency pattern]
    - Instructions: [any special directions]
    
    [continue for all medicines]
    
    IMPORTANT:
    - If there are conflicting interpretations, choose the most likely one based on frequency and confidence
    - If dosage or instructions are missing in some interpretations but present in others, include them
    - Maintain the exact format specified above
    - Don't include any explanations or notes - just the structured output
    """
    
    # Format all interpretations into a single string
    all_interpretations_text = "\n\n--- INTERPRETATION ---\n".join(all_interpretations)
    
    # Replace placeholder with actual interpretations
    consolidation_prompt = consolidation_prompt.replace("{INTERPRETATIONS}", all_interpretations_text)
    
    response = await client.aio.models.generate_content(
        model=model_id,
        contents=consolidation_prompt,
        config=GenerateContentConfig(
            temperature=0.1,
        ),
    )
    
    return response.text

# Async main
async def main():
    print("Running multiple interpretation passes in parallel...")
    start_time = asyncio.get_event_loop().time()

    all_interpretations = await run_parallel_interpretations()
    
    print("\nAll interpretation passes completed.\n")
    for idx, interpretation in enumerate(all_interpretations, 1):
        print(f"\n=== PASS {idx} ===\n{interpretation}\n{'-'*60}")

    print("\nGenerating consolidated final output...")
    final_result = await generate_final_output(all_interpretations)

    end_time = asyncio.get_event_loop().time()
    print(f"\nTotal processing time: {end_time - start_time:.2f} seconds")

    print("\n=== FINAL CONSOLIDATED PRESCRIPTION MEDICINES ===")
    print(final_result)

# Execute
await main()
