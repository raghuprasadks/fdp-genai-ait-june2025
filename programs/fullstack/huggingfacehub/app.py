from flask import Flask, request, jsonify
# from mistral import YourMistralModel  # Import your Mistral model or API client here
# Make sure hosted_mistralapi.py exists in the same directory or is installed as a package
from hostedmistralapi import get_response
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # This enables CORS for all routes

# Initialize your Mistral model or API client here
# model = YourMistralModel(...)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '')
    response_text=get_response(user_message)
    print("response_text:", response_text)
    # Call your Mistral model or API here
    # response_text = model.generate(user_message)

    #response_text = f"Echo: {user_message}"  # Replace with actual model call
    return jsonify({'response': response_text})

if __name__ == '__main__':
    app.run(debug=True)