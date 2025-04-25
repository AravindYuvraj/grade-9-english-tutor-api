from flask import Flask, request, jsonify
import os
from google import genai
from google.genai import types

app = Flask(__name__)

os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# Global variable to store the uploaded file URI (upload only once)
UPLOADED_FILE_URI = None
UPLOADED_FILE_MIME_TYPE = None

# Dictionary to store conversation histories for each user session
conversation_histories = {}

@app.route("/chat", methods=["POST"])
def chat():
    global UPLOADED_FILE_URI, UPLOADED_FILE_MIME_TYPE
    
    data = request.json
    question = data.get("question", "")
    session_id = data.get("session_id", "default")  # Use session ID to track different users
    
    # Initialize conversation history for this session if it doesn't exist
    if session_id not in conversation_histories:
        conversation_histories[session_id] = []
    
    # Upload the PDF file only once and cache it
    if UPLOADED_FILE_URI is None:
        uploaded_file = client.files.upload(file="AI_Studio/9th eng.pdf")
        UPLOADED_FILE_URI = uploaded_file.uri
        UPLOADED_FILE_MIME_TYPE = uploaded_file.mime_type
        
        # Add initial system message with PDF context
        initial_content = types.Content(
            role="user",
            parts=[
                types.Part.from_uri(
                    file_uri=UPLOADED_FILE_URI,
                    mime_type=UPLOADED_FILE_MIME_TYPE,
                ),
                types.Part.from_text(text="Let's start our tutoring session based on this material."),
            ],
        )
        conversation_histories[session_id].append(initial_content)
    
    # Add user's new message to conversation history
    user_content = types.Content(
        role="user",
        parts=[types.Part.from_text(text=question)],
    )
    conversation_histories[session_id].append(user_content)
    
    # Prepare the full conversation history
    contents = conversation_histories[session_id]
    
    # Configure the model
    generate_content_config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=0),  # Fast response mode
        response_mime_type="text/plain",
        system_instruction=[
            types.Part.from_text(text="""You are a friendly and supportive English tutor helping 9th grade students. 
            Keep your answers short, simple, and engaging. Your responses should be tailored to a 9th grade level.
            Remember previous parts of the conversation to maintain context.""")
        ]
    )

    # Stream the response
    output = ""
    try:
        for chunk in client.models.generate_content_stream(
            model="gemini-2.0-flash",  # Using the flash model for faster responses
            contents=contents,
            config=generate_content_config,
        ):
            if chunk.text:
                output += chunk.text
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
    # Add the model's response to the conversation history
    model_content = types.Content(
        role="model",
        parts=[types.Part.from_text(text=output)],
    )
    conversation_histories[session_id].append(model_content)
    
    # Clean up old conversations periodically (optional)
    # If a session gets too long, you might want to truncate it
    if len(conversation_histories[session_id]) > 20:  # Keep last 20 exchanges
        # Keep the PDF context message and the most recent messages
        conversation_histories[session_id] = [conversation_histories[session_id][0]] + conversation_histories[session_id][-19:]
    
    return jsonify({"response": output})

@app.route("/reset", methods=["POST"])
def reset_conversation():
    data = request.json
    session_id = data.get("session_id", "default")
    
    if session_id in conversation_histories:
        # Keep only the initial PDF context message
        initial_message = conversation_histories[session_id][0] if conversation_histories[session_id] else None
        conversation_histories[session_id] = [initial_message] if initial_message else []
    
    return jsonify({"status": "conversation reset"})

@app.route("/")
def home():
    return "Hello from your English Tutor API with conversation memory!"

# 🚀 Add this block to fix the Render port scan issue!
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)