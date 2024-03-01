from flask import Flask, request, render_template_string
from transformers import GPT2LMHeadModel, GPT2Tokenizer

app = Flask(__name__)

# Assuming your model and tokenizer are saved in the same directory with the name 'gpt2-viking'.
MODEL_PATH = "./gpt2-viking"

# Load model and tokenizer
model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)

def generate_text(prompt):
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(
        input_ids=inputs,
        max_length=80,
        temperature=0.1,
        top_k=2,
        top_p=0.95,
        repetition_penalty=3.0,
        do_sample=True,
        num_return_sequences=1
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Поговори с Рагнаром</title>
</head>
<body>
    <h1>Поговори с Рагнаром</h1>
    <form method="post">
        <textarea name="prompt" rows="4" cols="50" placeholder="Введите вопрос"></textarea>
        <br>
        <input type="submit" value="Сгенерировать">
    </form>
    {% if generated_text %}
        <h2>Ответ Рагнара</h2>
        <textarea rows="4" cols="50" readonly>{{ generated_text }}</textarea>
    {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    generated_text = ""
    if request.method == "POST":
        user_input = request.form["prompt"]
        generated_text = generate_text(user_input)
    return render_template_string(HTML_TEMPLATE, generated_text=generated_text)

if __name__ == "__main__":
    app.run(debug=True)