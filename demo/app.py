from flask import Flask, render_template, request, jsonify

import torch

from demo.models import (
    MLChatClassifier,
    BertChatClassifier,
    BartChatExplanator,
    BertMessageRegressor,
    decode_label
)

if torch.cuda.is_available():
    print(f"Using device: {torch.cuda.get_device_name(0)}")
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
    print("Using CPU")

app = Flask(__name__)
ml_clf = MLChatClassifier()
bert_clf = BertChatClassifier(DEVICE)
bart_expl = BartChatExplanator(DEVICE)
bert_reg = BertMessageRegressor(DEVICE)
# bart_expl_reg = BartChatRegressorExplanator(DEVICE)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/analysis/chat-classification', methods=['POST'])
def run_chat_classification():
    data = request.get_json()

    if not all(key in data for key in ['task', 'labels', 'preprocessor', 'model', 'messages']):
        return jsonify({'error': 'Missing required keys in request data'}), 400

    task = data['task']
    labels = data['labels']
    preprocessor = data['preprocessor']
    model = data['model']
    messages = data['messages']

    if len(messages) < 1:
        return jsonify({'error': 'At least 1 message required for ML analysis'}), 400

    if "bert" in model:
        label_id, probabilities = bert_clf.load_predict(
            task=task, labels=labels,
            model=model, msgs=messages
        )
    else:
        chat = [f"{msg['timestamp']} | {msg['person']}:\n{msg['message']}" for msg in messages]
        chat = "\n".join(chat)
        
        result = ml_clf.load_predict(
            task=task,
            labels=labels,
            preprocessor=preprocessor,
            model=model,
            chat=chat
        )
        if isinstance(result, tuple):
            label_id, probabilities = result
        else:
            label_id = result
            probabilities = []
    probabilities = [round(p, 2) for p in probabilities]
    return jsonify({
        'label': decode_label(label_id, labels),
        'probabilities': probabilities
    })

@app.route('/api/analysis/chat-explanation', methods=['POST'])
def run_chat_explanation():
    data = request.get_json()

    if not all(key in data for key in ['task', 'model', 'messages']):
        return jsonify({'error': 'Missing required keys in request data'}), 400

    task = data['task']
    model = data['model']
    messages = data['messages']

    if len(messages) < 1:
        return jsonify({'error': 'At least 2 messages required for BERT analysis'}), 400

    chat = [f"{msg['person']}:\n{msg['message']}" for msg in messages] # {msg['timestamp']} | 
    chat = "\n".join(chat)

    print("Generating explanation...")
    explanation = bart_expl.load_predict(
        task=task,
        model=model, text=chat
    )

    return jsonify({
        'output': explanation
    })

@app.route('/api/analysis/message-regression', methods=['POST'])
def run_message_regression():
    data = request.get_json()

    if not all(key in data for key in ['task', 'model', 'messages', 'target_idx']):
        return jsonify({'error': 'Missing required keys in request data'}), 400

    task = data['task']
    model = data['model']
    messages = data['messages']
    target_idx = data['target_idx']

    if len(messages) < 1:
        return jsonify({'error': 'At least 2 messages required for BERT analysis'}), 400

    output = bert_reg.load_predict(
        task=task, model=model,
        msgs=messages,
        target_idx=target_idx

    )

    return jsonify({
        'output': output[0].tolist()
    })

@app.route('/api/analysis/messages-regression-explanation', methods=['POST'])
def run_regression_explanation():
    data = request.get_json()

    if not all(key in data for key in ['task', 'labels', 'model', 'messages']):
        return jsonify({'error': 'Missing required keys in request data'}), 400

    task = data['task']
    labels = data['labels']
    model = data['model']
    messages = data['messages']

    if len(messages) < 1:
        return jsonify({'error': 'At least 2 messages required for BERT analysis'}), 400

    output = bart_expl_reg.load_predict(
        task=task, labels=labels,
        model=model, text=messages
    )

    return jsonify({
        'polarities': output['polarities'],
        'output': output['explanations']
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)