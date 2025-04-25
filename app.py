from flask import Flask, request, jsonify
import os
from google import genai
from google.genai import types
import traceback

app = Flask(__name__)

# === Configure API key ===
os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# === Upload the file ===
PDF_PATH = os.path.abspath("AI_Studio/9th eng.pdf")
uploaded_file = None

try:
    print(f"🔍 Trying to upload: {PDF_PATH}")
    uploaded_file = client.files.upload(file=PDF_PATH)
    print(f"✅ File uploaded: {uploaded_file.uri}")
except Exception as e:
    traceback.print_exc()
    print("❌ Failed to upload file.")
    uploaded_file = None

# === In-memory chat sessions ===
chat_sessions = {}

@app.route("/summary", methods=["POST"])
def get_summary():
    data = request.get_json()
    question = data.get("question", "")
    session_id = data.get("session_id", "default_user")

    if not question:
        return jsonify({"error": "Missing question"}), 400
    if uploaded_file is None:
        return jsonify({"error": "PDF file not uploaded to Gemini"}), 500

    # Initialize chat history
    if session_id not in chat_sessions:
        chat_sessions[session_id] = []

        # First message to Gemini
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

    # Add the current question
    chat_sessions[session_id].append(
        types.Content(
            role="user",
            parts=[types.Part.from_text(question)],
        )
    )

    # Trim chat history
    trimmed_history = chat_sessions[session_id][-10:]

    # Configuration for response
    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=0),
        response_mime_type="text/plain",
        system_instruction=[
            types.Part.from_text("You are a friendly and supportive English tutor helping 9th grade students. Keep your answers short and simple.")
        ]
    )

    # Generate the response
    response_text = ""
    try:
        for chunk in client.models.generate_content_stream(
            model="gemini-2.0-flash",
            contents=trimmed_history,
            config=config,
        ):
            response_text += chunk.text
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Gemini failed: {str(e)}"}), 500

    # Save the response
    chat_sessions[session_id].append(
        types.Content(
            role="model",
            parts=[types.Part.from_text(response_text)],
        )
    )

    return jsonify({"response": response_text})


@app.route("/")
def home():
    return "✅ English Tutor API is running with chat memory and fast response."


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
