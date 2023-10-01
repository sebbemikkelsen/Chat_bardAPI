from flask import Flask, render_template, request, jsonify
from taxing_law_chat import ask_bot

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    question = request.form.get('question')
    history = request.form.get('history')
    #response = ask_bot(question, history)
    response, filename = ask_bot(question, history, source=True)

    return jsonify({'response': response, 'filename': filename})

if __name__ == '__main__':
    app.run(debug=True)