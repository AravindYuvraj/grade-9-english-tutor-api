from flask import Flask, request, jsonify
import os
from google import genai
from google.genai import types

app = Flask(__name__)

# Securely get the API key from the environment variable
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set.")

client = genai.Client(api_key=GEMINI_API_KEY)

# Path to the PDF file within the Render instance
PDF_FILE_PATH = "data/9th eng.pdf"

# Keep track of the uploaded file URI (upload only once per instance lifecycle)
UPLOADED_FILE_URI = None
UPLOADED_FILE_MIME_TYPE = None

@app.before_first_request
def upload_pdf():
    global UPLOADED_FILE_URI, UPLOADED_FILE_MIME_TYPE
    try:
        with open(PDF_FILE_PATH, "rb") as f:
            uploaded_file = client.files.upload(file_data=f.read(), mime_type="application/pdf")
            UPLOADED_FILE_URI = uploaded_file.uri
            UPLOADED_FILE_MIME_TYPE = uploaded_file.mime_type
            print(f"File uploaded successfully on startup. URI: {UPLOADED_FILE_URI}")
    except FileNotFoundError:
        print(f"Error: PDF file not found at {PDF_FILE_PATH}")
    except Exception as e:
        print(f"Error uploading file on startup: {e}")


@app.route("/summary", methods=["POST"])
def get_summary():
    if UPLOADED_FILE_URI is None:
        return jsonify({"error": "File upload failed during application startup."}, 500)

    data = request.get_json()
    if not data or "question" not in data:
        return jsonify({"error": "Missing 'question' in the request body."}, 400)
    question = data.get("question")

    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_uri(
                    uri=UPLOADED_FILE_URI,
                    mime_type=UPLOADED_FILE_MIME_TYPE,
                ),
            ],
        ),
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=question),
            ],
        )
    ]

    generate_content_config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=0),
        response_mime_type="text/plain",
        system_instruction="You are a friendly and supportive English tutor helping 9th grade students. Keep your answers short and simple."
    )

    output = ""
    try:
        for chunk in client.models.generate_content_stream(
            model="gemini-2.0-flash",
            contents=contents,
            config=generate_content_config,
        ):
            output += chunk.text
        return jsonify({"response": output})
    except Exception as e:
        return jsonify({"error": f"Error generating summary: {e}"}, 500)


@app.route("/")
def home():
    return "Hello from your English Tutor API!"

if __name__ == "__main__":
    app.run(debug=True)