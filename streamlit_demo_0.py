import os
import streamlit as st
from google import genai
from google.genai.types import Part, Tool, GenerateContentConfig, GoogleSearch
import concurrent.futures
import time
import nest_asyncio
import re
import difflib
from collections import defaultdict
from dotenv import load_dotenv


# Load environment variables
load_dotenv()

# Apply nest_asyncio to allow nested event loops (needed for Streamlit)
nest_asyncio.apply()

# Streamlit page configuration
st.set_page_config(page_title="Prescription Reader", page_icon="💊", layout="wide")

# Initialize the client with API key from Streamlit secrets
try:
    API_KEY = os.getenv("GOOGLE_API_KEY")
except KeyError:
    st.error("Please set the GOOGLE_API_KEY in .streamlit/secrets.toml")
    st.stop()

client = genai.Client(api_key=API_KEY)
model_id = "gemini-2.0-flash"

# Configure Google Search Tool
google_search_tool = Tool(google_search=GoogleSearch())

# Streamlit UI
st.title("Prescription Reader")
st.markdown(
    "Upload a handwritten prescription image to extract and verify medicine names."
)

# File uploader
uploaded_file = st.file_uploader(
    "Choose a prescription image (JPEG)", type=["jpg", "jpeg"]
)


# Function to make a single API call for interpretation
def generate_interpretation(temperature, image):
    initial_prompt = """
    This image contains a handwritten prescription. Please analyze it and provide:
    1. Your interpretation of each medicine name in the prescription
    2. For each medicine name, assign a confidence percentage (0-100%)
    3. Format each line as "Medicine Name: [confidence]%"

    Only focus on identifying medicine names, not other text in the image.
    List exactly the number of medicines you see in the prescription - no more, no less.
    Medicine name should be in a list like:
    1. --
    2. --
    """
    response = client.models.generate_content(
        model=model_id,
        contents=[initial_prompt, image],
        config=GenerateContentConfig(temperature=temperature),
    )
    return response.text


# Run all interpretations in parallel using ThreadPoolExecutor
def run_parallel_interpretations(image, num_passes=5):
    temperatures = [0.7 + (i * 0.2) for i in range(num_passes)]

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_passes) as executor:
        future_to_temp = {
            executor.submit(generate_interpretation, temp, image): temp
            for temp in temperatures
        }
        results = []
        for future in concurrent.futures.as_completed(future_to_temp):
            temp = future_to_temp[future]
            try:
                result = future.result()
                st.write(f"Completed interpretation with temperature {temp:.1f}")
                results.append((temp, result))
            except Exception as e:
                st.error(f"Error processing temperature {temp}: {e}")

    results.sort()
    return [r[1] for r in results]


# Function to extract a list of possible medicine names from all interpretations
def extract_medicine_candidates(interpretations):
    medicine_candidates = []

    for interpretation in interpretations:
        lines = interpretation.strip().split("\n")
        for line in lines:
            match = re.search(r"^\d+\.\s+(.*?):\s*(\d+)%", line)
            if match:
                medicine_name = match.group(1).strip()
                confidence = int(match.group(2))
                position_match = re.search(r"^(\d+)\.", line)
                position = int(position_match.group(1)) if position_match else 0

                medicine_candidates.append(
                    {
                        "name": medicine_name,
                        "confidence": confidence,
                        "position": position,
                    }
                )

    return medicine_candidates


# Function to group similar medicine name candidates
def group_similar_medicines(medicine_candidates):
    medicine_candidates.sort(key=lambda x: x["position"])
    position_groups = defaultdict(list)
    for candidate in medicine_candidates:
        position_groups[candidate["position"]].append(candidate)

    formatted_groups = []
    for position in sorted(position_groups.keys()):
        group = position_groups[position]
        group_text = f"Medicine {position}: ["
        medicine_texts = [f"{med['name']}: {med['confidence']}%" for med in group]
        group_text += ", ".join(medicine_texts) + "]"
        formatted_groups.append((position, group_text, group))

    return formatted_groups


# Function to verify grouped medicines using Google Search
def verify_medicine_groups(medicine_groups):
    verification_prompts = []

    for position, group_text, group in medicine_groups:
        prompt = f"""
        I have multiple interpretations of a medicine name from a handwritten prescription (position #{position}):
        
        {group_text}
        
        Please analyze these interpretations and determine the most likely correct medicine name.
        Focus on medicines available in Bangladesh and check pharmaceutical websites like MedEx or Arogga or  Lazzpharma or MedEasy.
        
        Respond with:
        1. The correct medicine name (taken from Medex or Arogga or Lazzpharma or MedEasy)
        2. The dosage information if available
        3. Brief description of what this medicine is used for
        
        Important: To get better search result carefully choose the medicine name from the group
        """
        verification_prompts.append((position, prompt))

    verification_results = []

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(6, len(verification_prompts))
    ) as executor:
        future_to_position = {
            executor.submit(
                lambda p: (
                    p[0],
                    client.models.generate_content(
                        model=model_id,
                        contents=p[1],
                        config=GenerateContentConfig(
                            tools=[google_search_tool],
                            response_modalities=["TEXT"],
                            temperature=0.2,
                        ),
                    ).text,
                ),
                prompt,
            ): prompt
            for prompt in verification_prompts
        }

        for future in concurrent.futures.as_completed(future_to_position):
            prompt = future_to_position[future]
            try:
                position, result = future.result()
                verification_results.append((position, result))
                st.write(f"Verified medicine at position {position}")
            except Exception as e:
                st.error(f"Error verifying medicine at position {prompt[0]}: {e}")
                verification_results.append((prompt[0], f"Error: {str(e)}"))

    verification_results.sort(key=lambda x: x[0])
    return verification_results


# Function to process final results and format the output
def format_final_results(verification_results):
    final_prompt = """
    You are a medical prescription expert specializing in Bangladeshi medicines. 
    I will provide you with verification results for medicines from a prescription.
    
    For each medicine, create a clean, formatted entry with:
    1. The medicine name (exactly the most correct one, based on the verification)
    2. The dosage information
    3. Any instructions for taking the medicine
    
    Format your response as:
    ```
    FINAL PRESCRIPTION MEDICINES:
    
    1. Medicine Name: 
       Dosage: [dosage info]
       Instructions: [any special instructions]
    
    2. Medicine Name: 
       Dosage: [dosage info]
       Instructions: [any special instructions]
    
    [continue for all medicines in the prescription]
    ```
    
    Here are the verification results for each medicine position:
    
    {VERIFICATION_RESULTS}
    """

    formatted_results = ""
    for i, (position, result) in enumerate(verification_results, 1):
        formatted_results += f"\n--- Medicine Position {position} ---\n"
        formatted_results += result + "\n"
        formatted_results += "-" * 40 + "\n"

    final_prompt = final_prompt.replace("{VERIFICATION_RESULTS}", formatted_results)

    response = client.models.generate_content(
        model=model_id,
        contents=final_prompt,
        config=GenerateContentConfig(
            tools=[google_search_tool],
            response_modalities=["TEXT"],
            temperature=0.1,
        ),
    )

    return response.text


# Main execution flow
def main():
    if uploaded_file is None:
        st.warning("Please upload an image to proceed.")
        return

    # Convert uploaded file to Part for Gemini API
    image = Part.from_bytes(data=uploaded_file.read(), mime_type="image/jpeg")

    # Process button
    if st.button("Process Prescription"):
        with st.spinner("Processing prescription..."):
            start_time = time.time()

            # Execute all interpretation passes
            st.write("Running multiple interpretation passes in parallel...")
            all_interpretations = run_parallel_interpretations(image)

            interpretation_time = time.time()
            st.write(
                f"All interpretation passes completed in {interpretation_time - start_time:.2f} seconds"
            )

            # Display interpretations
            with st.expander("View Interpretation Passes"):
                for i, interpretation in enumerate(all_interpretations, 1):
                    st.write(f"### Pass {i} Interpretation")
                    st.code(interpretation)

            # Extract medicine candidates
            medicine_candidates = extract_medicine_candidates(all_interpretations)
            st.write(f"Extracted {len(medicine_candidates)} medicine candidates")

            # Check if any candidates were found
            if not medicine_candidates:
                st.error("No medicines were identified in the prescription.")
                final_result = "```\nFINAL PRESCRIPTION MEDICINES:\n\nNo medicines were identified in the prescription.\n```"
                st.write("### Final Prescription Medicines")
                st.code(final_result)
                end_time = time.time()
                st.write(f"Total processing time: {end_time - start_time:.2f} seconds")
                return

            # Group similar medicines
            medicine_groups = group_similar_medicines(medicine_candidates)
            st.write(f"Grouped into {len(medicine_groups)} medicine positions")

            # Display groups
            with st.expander("View Medicine Interpretation Groups"):
                for position, group_text, _ in medicine_groups:
                    st.write(group_text)

            # Verify medicine groups
            st.write(
                "Verifying medicine groups with Google Search (this may take some time)..."
            )
            verification_results = verify_medicine_groups(medicine_groups)

            verification_time = time.time()
            st.write(
                f"Medicine verification completed in {verification_time - interpretation_time:.2f} seconds"
            )

            # Format final results
            st.write("Generating final analysis with verified medicine names...")
            final_result = format_final_results(verification_results)

            end_time = time.time()
            st.write(
                f"Final analysis completed in {end_time - verification_time:.2f} seconds"
            )
            st.write(f"Total processing time: {end_time - start_time:.2f} seconds")

            # Display final results
            st.write("### Final Prescription Medicines")
            st.code(final_result)


if __name__ == "__main__":
    main()
