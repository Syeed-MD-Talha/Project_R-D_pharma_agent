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
API_KEY = "AIzaSyBpDTvRmjhbk9ryLibwaXS2HMKJPBAYsmY"
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
    
    OUTPUT FORMAT:
    ```
    EXTRACTED MEDICINES:
    
    Medicine 1:
    - Name: [exactly as written (don't include the power just medicine name only)]
    - Dosage: [frequency pattern]
    - Instructions: [any special directions]
    
    Medicine 2:
    - Name: [exactly as written]
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
            temperature=temperature,
        ),
    )
    return response.text

# Run multiple interpretations concurrently
async def run_parallel_interpretations(num_passes=3):
    # Using fixed temperature of 0.2 as in your updated paste
    # But running multiple passes for different interpretations
    tasks = [generate_interpretation(0.2) for _ in range(num_passes)]
    results = await asyncio.gather(*tasks)
    return results

# Generate final consolidated output
async def generate_consolidated_medicines(all_interpretations):
    """First consolidate all interpretations into a single list of medicines"""
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

async def verify_medicine_name(medicine_name):
    """Verify a medicine name using Google Search, focusing on Bangladesh pharmacy websites"""
    verification_prompt = f"""
    I need to verify a medicine name '{medicine_name}' that was extracted from a handwritten prescription.
    
    Please search for this medicine name on Bangladeshi pharmacy websites like MedEx, Arogga, or Lazzypharm.
    
    If you find a close match (e.g., "Amlovand" is actually "Amlocard"), provide:
    1. The correct medicine name as found on these websites (format: OriginalName → VerifiedName)
    2. A brief description of what this medicine is used for
    3. The pharmaceutical company that makes it (if available)
    4. URL of the medicine page (if available)
    
    If you can't find a close match, just indicate that you couldn't verify it.
    
    IMPORTANT: 
    1. Focus your search specifically on Bangladeshi pharmaceutical websites and databases
    2. Use search terms like "[medicine name] bangladesh pharmacy" or "[medicine name] medex arogga"
    3. Make sure to check for spelling variations and similar-sounding medicine names
    4. Be concise in your response
    """
    
    try:
        response = await client.aio.models.generate_content(
            model=model_id,
            contents=verification_prompt,
            config=GenerateContentConfig(
                tools=[google_search_tool],
                temperature=0.1,
            ),
        )
        return response.text
    except Exception as e:
        return f"Error verifying medicine: {str(e)}"

async def extract_medicines_from_consolidated(consolidated_text):
    """Extract medicine names from the consolidated text for verification"""
    # This regex looks for "Medicine X:" followed by a "- Name:" line
    pattern = r"Medicine \d+:\s*\n\s*- Name: ([^\n\(]+)"
    medicine_names = re.findall(pattern, consolidated_text)
    return [name.strip() for name in medicine_names]

async def generate_final_output(all_interpretations):
    """Generate final output with verified medicine names"""
    # First consolidate all interpretations
    consolidated_text = await generate_consolidated_medicines(all_interpretations)
    
    # Extract medicine names from consolidated text
    medicine_names = await extract_medicines_from_consolidated(consolidated_text)
    
    print(f"\nFound {len(medicine_names)} medicines to verify...")
    print(f"Starting parallel verification of all medicines...")
    
    # Verify all medicine names in parallel
    verification_tasks = [verify_medicine_name(name) for name in medicine_names]
    verification_results_list = await asyncio.gather(*verification_tasks)
    
    # Create a dictionary mapping medicine names to their verification results
    verification_results = {name: result for name, result in zip(medicine_names, verification_results_list)}
    
    for name, result in verification_results.items():
        print(f"Verified medicine: {name}")
    
    # Create final output with verification results
    final_prompt = f"""
    Here is a consolidated list of medicines from a prescription:
    
    {consolidated_text}
    
    For each medicine, I have verification results from searching Bangladeshi pharmacy websites:
    
    {verification_results}
    
    Please create a final formatted output that includes:
    1. The original medicine name from the prescription
    2. The verified correct medicine name from pharmacy websites (if found)
    3. Dosage and instructions from the original consolidated list
    4. A brief description of what the medicine is used for (if available)
    
    Use this exact format:
    
    Medicine 1:
    - Name: [original handwritten name] → [verified name from pharmacy websites, if different]
    - Dosage: [frequency pattern]
    - Instructions: [any special directions]
    - Description: [brief description of what the medicine is used for]
    
    Medicine 2:
    ...
    
    IMPORTANT: 
    - If the verified name is the same as the original, just show the name once
    - If no verified name was found, only show the original name
    - Keep the same order of medicines as in the consolidated list
    """
    
    response = await client.aio.models.generate_content(
        model=model_id,
        contents=final_prompt,
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

    print("\nConsolidating interpretations and verifying medicine names...")
    final_result = await generate_final_output(all_interpretations)

    end_time = asyncio.get_event_loop().time()
    print(f"\nTotal processing time: {end_time - start_time:.2f} seconds")

    print("\n=== FINAL PRESCRIPTION MEDICINES WITH VERIFICATION ===")
    print(final_result)

# Execute
await main()
