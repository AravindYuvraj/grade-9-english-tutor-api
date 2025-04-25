from flask import Flask, request, jsonify
import os
from google import genai
from google.genai import types

app = Flask(__name__)

# Initialize Gemini
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise EnvironmentError("GEMINI_API_KEY environment variable not set.")
client = genai.Client(api_key=api_key)

# Upload file once at startup
PDF_PATH = "AI_Studio/9th eng.pdf"
if not os.path.exists(PDF_PATH):
    raise FileNotFoundError(f"PDF file not found at {PDF_PATH}")
uploaded_file = client.files.upload(file=PDF_PATH)

# In-memory chat storage (can be replaced with DB later)
chat_sessions = {}

@app.route("/summary", methods=["POST"])
def get_summary():
    try:
        data = request.get_json()
        question = data.get("question", "")
        session_id = data.get("session_id", "default_user")

        if not question:
            return jsonify({"error": "Missing question"}), 400

        # Initialize chat session if not exists
        if session_id not in chat_sessions:
            chat_sessions[session_id] = []

            # Start with file + intro message
            chat_sessions[session_id].append(
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_uri(
                            file_uri=uploaded_file.uri,
                            mime_type=uploaded_file.mime_type,
                        ),
                        types.Part.from_text("Hi"),
                    ],
                )
            )
            chat_sessions[session_id].append(
                types.Content(
                    role="model",
                    parts=[
                        types.Part.from_text("Hi! How can I help you with your English lessons?")
                    ]
                )
            )

        # Add the new user message
        chat_sessions[session_id].append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(question)],
            )
        )

        # Limit history to last 10 messages (5 turns)
        trimmed_history = chat_sessions[session_id][-10:]

        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            response_mime_type="text/plain",
            system_instruction=[
                types.Part.from_text("You are a friendly and supportive English tutor helping 9th grade students. Keep your answers short and simple.")
            ]
        )

        response_text = ""
        for chunk in client.models.generate_content_stream(
            model="gemini-2.0-flash",
            contents=trimmed_history,
            config=config,
        ):
            response_text += chunk.text

        # Save Gemini's reply into memory for next turn
        chat_sessions[session_id].append(
            types.Content(
                role="model",
                parts=[types.Part.from_text(response_text)],
            )
        )

        return jsonify({"response": response_text})

    except Exception as e:
        # Log the exception
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/")
def home():
    return "✅ English Tutor API is running with chat memory and fast response."

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)
