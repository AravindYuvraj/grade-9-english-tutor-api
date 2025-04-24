from flask import Flask, request, jsonify
import os
from google import genai
from google.genai import types

app = Flask(__name__)

os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

@app.route("/summary", methods=["POST"])
def get_summary():
    data = request.json
    question = data.get("question", "")

    # Upload file once (optional optimization: move to init if already uploaded)
    uploaded_file = client.files.upload(file="AI_Studio/9th eng.pdf")

    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_uri(
                    file_uri=uploaded_file.uri,
                    mime_type=uploaded_file.mime_type,
                ),
                types.Part.from_text(text="hi"),
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
        system_instruction=[
            types.Part.from_text(text="""You are a friendly and supportive English tutor helping 9th grade students. Keep your answers short and simple.""")
        ]
    )

    output = ""
    for chunk in client.models.generate_content_stream(
        model="gemini-2.0-flash",
        contents=contents,
        config=generate_content_config,
    ):
        output += chunk.text

    return jsonify({"response": output})


@app.route("/")
def home():
    return "Hello from your English Tutor API!"

