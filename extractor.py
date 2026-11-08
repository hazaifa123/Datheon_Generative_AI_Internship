import os
import json
import logging
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

# Setup clean console logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Schema definition for candidate data
class ResumeSchema(BaseModel):
    full_name: str = Field(description="Full name of the candidate")
    email: str = Field(description="Candidate email address")
    skills: list[str] = Field(description="Extracted technical and professional skills")
    summary: str = Field(description="A concise 2-sentence professional summary")


def get_gemini_client() -> genai.Client:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is missing. Please set it in your .env file.")
    return genai.Client(api_key=api_key)


def extract_resume_data(client: genai.Client, raw_text: str) -> ResumeSchema:
    prompt = (
        "Extract the candidate's profile information from the following text "
        "and return strictly according to the schema.\n\n"
        f"Text:\n{raw_text}"
    )

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ResumeSchema,
            temperature=0.1,  # Kept low for deterministic extraction
        ),
    )
    
    # Parse into Pydantic model for validation
    return ResumeSchema.model_validate_json(response.text)


def main():
    sample_cv = """
    M. Hazaifa
    Software Engineer based in Lahore, Pakistan. Specializing in Python development,
    LangChain, and building AI agents. Familiar with Docker, Supabase, and Flutter.
    Contact me at hazaifa@example.com for collaboration.
    """

    try:
        logging.info("Initializing Gemini client...")
        client = get_gemini_client()

        logging.info("Processing unstructured text via Gemini...")
        extracted_data = extract_resume_data(client, sample_cv)

        # Convert to dictionary and pretty JSON
        result_dict = extracted_data.model_dump()
        output_filepath = "output.json"

        with open(output_filepath, "w", encoding="utf-8") as f:
            json.dump(result_dict, f, indent=4)

        logging.info("Data extracted successfully!")
        print("\n--- Extracted JSON Output ---")
        print(json.dumps(result_dict, indent=4))
        print(f"\nResult saved to {output_filepath}")

    except Exception as e:
        logging.error(f"Execution failed: {e}")


if __name__ == "__main__":
    main()