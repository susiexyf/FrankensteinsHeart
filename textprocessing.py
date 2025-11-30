import sys
sys.modules["scipy"] = None
sys.modules["scipy.stats"] = None
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('vader_lexicon')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer

INPUT_FILE = "frankenstein.txt"
OUTPUT_HTML = "output.html"

# Filler words remain as text
filler_words = set(stopwords.words("english"))
filler_words.update(["the", "an", "a"])  


def sentiment_to_emoji(score):
    """
    Map VADER sentiment compound score (-1.0 to +1.0) to colored heart emojis.
    """

    if score > 0.75:
        return "🩷"   # extremely positive
    elif score > 0.50:
        return "❤️"   # strong positive
    elif score > 0.30:
        return "🧡"   # warm positive
    elif score > 0.15:
        return "💛"   # moderately positive
    elif score > 0.05:
        return "💚"   # slightly positive
    elif score > -0.05:
        return "🤍"   # neutral
    elif score > -0.15:
        return "💙"   # slightly negative / introspective
    elif score > -0.30:
        return "💜"   # moderately negative / sad
    elif score > -0.50:
        return "🤎"   # mild negative
    elif score > -0.75:
        return "🖤"   # strong negative
    else:
        return "💔"   # extremely negative


def process_text():
    # Load input text
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        text = f.read()

    tokens = word_tokenize(text)
    sia = SentimentIntensityAnalyzer()

    result = []

    for token in tokens:
        clean = token.lower()

        # Leave filler words and punctuation alone
        if clean in filler_words or not clean.isalpha():
            result.append(token)
            continue

        # Analyze sentiment for content words
        score = sia.polarity_scores(clean)["compound"]

        # Replace word with emoji
        emoji = sentiment_to_emoji(score)
        result.append(emoji)

    return result


def build_html(tokens):
    # Join tokens with spaces, safe for HTML
    body = " ".join(tokens)

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Emoji Sentiment Output</title>
        <style>
            body {{
                font-family: Georgia, serif;
                margin: 40px;
                font-size: 22px;
                line-height: 1.6;
            }}
        </style>
    </head>
    <body>
        {body}
    </body>
    </html>
    """

    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)

    print("HTML created:", OUTPUT_HTML)


if __name__ == "__main__":
    tokens = process_text()
    build_html(tokens)
