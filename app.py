import streamlit as st
from transformers import MarianMTModel, MarianTokenizer

# Function to load the translation model and tokenizer
def load_model(model_name):
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    return model, tokenizer

# Streamlit App
st.title("Language Translation App For Engineer Zahid Astori ")
st.subheader("Translate mulittple languages instantly ")

# Available languages
languages = {
    'Azerbaijani': 'az',
    'English': 'en',
    'French': 'fr',
    'German': 'de',
    'Spanish': 'es',
    'Italian': 'it',
    'Portuguese': 'pt',
}

# User input for language selection
input_lang = st.selectbox("Select Input Language:", list(languages.keys()))
output_lang = st.selectbox("Select Output Language:", list(languages.keys()))

# Ensure input and output languages are not the same
if input_lang == output_lang:
    st.error("Input and output languages must be different.")
else:
    # Prepare model name
    model_name = f"Helsinki-NLP/opus-mt-{languages[input_lang]}-{languages[output_lang]}"

    try:
        # Load model and tokenizer
        model, tokenizer = load_model(model_name)

        # User input for text to translate
        text_to_translate = st.text_area("Enter text to translate:")

        # Translation functionality
        if st.button("Translate"):
            if text_to_translate:
                # Tokenize and translate the input text
                inputs = tokenizer(text_to_translate, return_tensors="pt", padding=True)
                translated = model.generate(**inputs)
                translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)

                # Show the result
                st.subheader("Translated Text:")
                st.write(translated_text)
            else:
                st.error("Please enter some text to translate.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
