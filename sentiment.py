from transformers import pipeline

def analyze_sentiment(titles):
    try:
        # Load BERT for sentiment analysis (lightweight for your setup)
        classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", tokenizer="distilbert-base-uncased-finetuned-sst-2-english")
        sentiments = [classifier(title)[0]['label'] for title in titles[:5]]  # Top 5 titles
        positive = sum(1 for s in sentiments if s == "POSITIVE")
        negative = sum(1 for s in sentiments if s == "NEGATIVE")
        return "Positive" if positive > negative else "Negative" if negative > positive else "Neutral"
    except Exception as e:
        print(f"Error analyzing sentiment: {e}")
        return "Neutral"