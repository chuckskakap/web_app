from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import pickle
import keras_nlp

app = Flask(__name__)

MAX_SEQUENCE_LENGTH = 40

# Load the model
loaded_model = tf.keras.models.load_model("transformer_model")

# Load jw_vocab
with open("jw_vocab.pkl", "rb") as f:
    jw_vocab = pickle.load(f)

# Load rm_vocab
with open("rm_vocab.pkl", "rb") as f:
    rm_vocab = pickle.load(f)

# Load tokenizers
jw_tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
    vocabulary=jw_vocab, lowercase=False
)
rm_tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
    vocabulary=rm_vocab, lowercase=False
)

# Update logo_url in app.py
logo_url = "/static/logo.png"


@app.route("/")
def home():
    return render_template("index.html", logo_url=logo_url)


@app.route("/translate", methods=["POST"])
def translate():
    input_text = request.form["input_text"]
    translated_text = translate_text(input_text)
    return render_template(
        "index.html", input_text=input_text, translated_text=translated_text
    )


def translate_text(input_text):
    # Tokenize the input sentence
    input_tokens = jw_tokenizer([input_text]).to_tensor(
        shape=(None, MAX_SEQUENCE_LENGTH)
    )

    # Generate translation
    translated_tokens = keras_nlp.samplers.GreedySampler()(
        lambda prompt, cache, index: (
            loaded_model([input_tokens, prompt])[:, index - 1, :],
            None,
            cache,
        ),
        prompt=tf.concat(
            [
                tf.fill(
                    (tf.shape(input_tokens)[0], 1), rm_tokenizer.token_to_id("[START]")
                ),
                tf.fill(
                    (tf.shape(input_tokens)[0], MAX_SEQUENCE_LENGTH - 1),
                    rm_tokenizer.token_to_id("[PAD]"),
                ),
            ],
            axis=-1,
        ),
        end_token_id=rm_tokenizer.token_to_id("[END]"),
        index=1,
    )

    # Detokenize the translated tokens
    translated_text = rm_tokenizer.detokenize(translated_tokens)

    return (
        translated_text.numpy()[0]
        .decode("utf-8")
        .replace("[PAD]", "")
        .replace("[START]", "")
        .replace("[END]", "")
        .strip()
    )


if __name__ == "__main__":
    app.run(debug=True)
