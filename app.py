from flask import Flask, request, jsonify
import os
import google.generativeai as genai
import inspect

app = Flask(__name__)

# Securely get the API key from the environment variable
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set.")

genai.configure(api_key=GEMINI_API_KEY)

# Path to the PDF file within the Render instance
PDF_FILE_PATH = "data/9th eng.pdf"

# Keep track of the uploaded file URI (upload only once per instance lifecycle)
UPLOADED_FILE_URI = None
UPLOADED_FILE_MIME_TYPE = None

def upload_pdf():
    global UPLOADED_FILE_URI, UPLOADED_FILE_MIME_TYPE
    try:
        # Debug: Check if file exists and print current directory contents
        print(f"Current directory: {os.getcwd()}")
        print(f"Files in current directory: {os.listdir('.')}")
        if os.path.exists("data"):
            print(f"Files in data directory: {os.listdir('data')}")
        else:
            print("data directory not found")
            
        # Debug: Print function signature to see required parameters
        print(f"Upload file function signature: {inspect.signature(genai.upload_file)}")
        
        # Check if file exists
        if not os.path.exists(PDF_FILE_PATH):
            print(f"Error: PDF file not found at {PDF_FILE_PATH}")
            return
            
        # Read file data
        with open(PDF_FILE_PATH, "rb") as f:
            file_data = f.read()
        
        # Try upload with minimal parameters
        try:
            # Current API (as of April 2025) likely uses this format
            uploaded_file = genai.upload_file(file_data, mime_type="application/pdf")
            UPLOADED_FILE_URI = uploaded_file.uri
            UPLOADED_FILE_MIME_TYPE = "application/pdf"
            print(f"File uploaded successfully on startup. URI: {UPLOADED_FILE_URI}")
        except Exception as e:
            print(f"Error during file upload attempt: {e}")
            print("Skipping file upload for now. Service will run without PDF functionality.")
    except Exception as e:
        print(f"Error in upload_pdf function: {e}")

# Replace before_first_request with an app setup function
@app.before_request
def before_request_func():
    global UPLOADED_FILE_URI
    if UPLOADED_FILE_URI is None:
        upload_pdf()


@app.route("/summary", methods=["POST"])
def get_summary():
    if UPLOADED_FILE_URI is None:
        return jsonify({"error": "File upload failed during application startup."}), 500

    data = request.get_json()
    if not data or "question" not in data:
        return jsonify({"error": "Missing 'question' in the request body."}), 400
    question = data.get("question")

    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Create a list of contents for the model
        contents = [
            {
                "role": "user",
                "parts": [
                    {"file_data": {
                        "mime_type": UPLOADED_FILE_MIME_TYPE,
                        "file_uri": UPLOADED_FILE_URI
                    }},
                    {"text": question}
                ]
            }
        ]
        
        response = model.generate_content(
            contents=contents,
            stream=True
        )
        
        output = ""
        for chunk in response:
            if hasattr(chunk, 'text'):
                output += chunk.text
        
        return jsonify({"response": output})
    except Exception as e:
        print(f"Error generating summary: {e}")
        return jsonify({"error": f"Error generating summary: {e}"}), 500


@app.route("/")
def home():
    return "Hello from your English Tutor API!"

if __name__ == "__main__":
    # The first request will trigger the PDF upload
    app.run(debug=True)