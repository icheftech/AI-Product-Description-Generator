import os
import logging

from flask import Flask, render_template, request, flash
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

# -------------------------------------------------------------------
# Flask app configuration
# -------------------------------------------------------------------
app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", "change-me-in-production")

# -------------------------------------------------------------------
# OpenAI client configuration
# -------------------------------------------------------------------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    # We won't crash the app, but we will log a clear error.
    logging.warning("OPENAI_API_KEY is not set. The app will run, but API calls will fail.")

client = OpenAI(api_key=OPENAI_API_KEY)

# -------------------------------------------------------------------
# Tone configuration
# -------------------------------------------------------------------
TONE_STYLES = {
    "casual": {
        "label": "Casual",
        "instruction": (
            "Write in a friendly, conversational tone, like you're talking to a friend. "
            "Use simple language and a relaxed style."
        ),
    },
    "professional": {
        "label": "Professional",
        "instruction": (
            "Write in a clear, concise, and professional tone suitable for a corporate audience. "
            "Avoid slang and keep it polished."
        ),
    },
    "enthusiastic": {
        "label": "Enthusiastic",
        "instruction": (
            "Write in an energetic and upbeat tone that builds excitement. "
            "Use strong, positive language that makes the product feel irresistible."
        ),
    },
    "technical": {
        "label": "Technical",
        "instruction": (
            "Write in a technical, detail-oriented tone. "
            "Emphasize specifications, performance, and measurable benefits."
        ),
    },
    "luxury": {
        "label": "Luxury",
        "instruction": (
            "Write in a premium, luxurious tone. "
            "Highlight exclusivity, craftsmanship, and high-end lifestyle benefits."
        ),
    },
}


# -------------------------------------------------------------------
# Helper function to call OpenAI
# -------------------------------------------------------------------
def generate_product_description(product_name: str, keywords: str, tone_key: str) -> str:
    """
    Call OpenAI to generate a product description using the selected tone.
    """
    tone = TONE_STYLES.get(tone_key, TONE_STYLES["professional"])
    tone_instruction = tone["instruction"]

    # Normalize and format keywords
    keyword_list = [k.strip() for k in keywords.split(",") if k.strip()]
    keyword_str = ", ".join(keyword_list) if keyword_list else "None"

    system_message = (
        "You are an expert retail copywriter specializing in writing product descriptions "
        "that increase conversions. Always write in plain text (no markdown). "
        "Write 1–3 short paragraphs and optionally 3–5 bullet-style phrases separated by commas "
        "for key benefits. Keep the description between 120 and 220 words."
    )

    user_message = (
        f"Product name: {product_name}\n"
        f"Tone: {tone['label']}\n"
        f"Tone Instructions: {tone_instruction}\n"
        f"Keywords to incorporate naturally: {keyword_str}\n\n"
        "Write a compelling product description that:\n"
        "- Matches the tone described above\n"
        "- Naturally weaves in the keywords without sounding forced\n"
        "- Focuses on benefits, not just features\n"
        "- Speaks directly to retail customers shopping online\n"
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
        temperature=0.8,
        max_tokens=300,
    )

    return response.choices[0].message.content.strip()


# -------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    generated_text = None

    # Preserve form values on errors
    form_product_name = ""
    form_keywords = ""
    form_tone = "professional"

    if request.method == "POST":
        form_product_name = request.form.get("product_name", "").strip()
        form_keywords = request.form.get("keywords", "").strip()
        form_tone = request.form.get("tone", "professional")

        # Basic validation
        if not form_product_name:
            flash("Please enter a product name.", "warning")
        elif not OPENAI_API_KEY:
            flash(
                "OpenAI API key is not configured. Please set OPENAI_API_KEY and restart the app.",
                "danger",
            )
        else:
            try:
                generated_text = generate_product_description(
                    product_name=form_product_name,
                    keywords=form_keywords,
                    tone_key=form_tone,
                )
            except Exception as e:
                # Log detailed error, show generic message to user
                logging.exception("Error while generating product description: %s", e)
                flash(
                    "Sorry, something went wrong while generating the description. "
                    "Please try again.",
                    "danger",
                )

    return render_template(
        "index.html",
        tones=TONE_STYLES,
        generated_text=generated_text,
        product_name=form_product_name,
        keywords=form_keywords,
        selected_tone=form_tone,
    )


# -------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------
if __name__ == "__main__":
    # For development only. In production, use a WSGI server (e.g., gunicorn).
    app.run(host="0.0.0.0", port=5000, debug=True)
