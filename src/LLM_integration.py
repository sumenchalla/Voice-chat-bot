from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os
from gtts import gTTS
from src.Speach_recognition import results
from utlies.support import vb_collection


# Load tokenizer and model
model_name = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


question =results.text
context = vb_collection.query(
    query_texts=f"{question}",
    n_results=1)
input_text = f"""
    Based on the below data,
    answer this question {question}.
    and the data is {context["documents"]}
"""

# Tokenize input text
inputs = tokenizer(input_text, max_length=1024, return_tensors="pt", truncation=True)

# Generate summary
summary_ids = model.generate(inputs["input_ids"], max_length=150, min_length=40, num_beams=4, early_stopping=True)

# Decode and print the summary
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
myobj = gTTS(text=summary, lang="en", slow=False)
