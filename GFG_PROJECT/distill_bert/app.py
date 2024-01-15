from flask import Flask, render_template, request
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, GPT2Tokenizer, GPT2LMHeadModel
from transformers import pipeline, DistilBertTokenizer, DistilBertForQuestionAnswering


app = Flask(__name__)

# Load the DistilBERT model for classification
# Load the DistilBERT model and tokenizer
tokenizer_distilbert = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model_distilbert = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')

# Load the saved DistilBERT model parameters with map_location set to 'cpu'
checkpoint_distilbert = torch.load('distill_bert\distilbert_model.pth', map_location=torch.device('cpu'))

model_distilbert.load_state_dict(checkpoint_distilbert['model_state_dict'])
model_distilbert.eval()

# Set the DistilBERT model to evaluation mode






def detect_depression(input_text, model, tokenizer):
    input_encoding = tokenizer(input_text, return_tensors='pt')
    # input_encoding = {key: value.to(device) for key, value in input_encoding.items()}
    output = model(**input_encoding)
    probability = torch.sigmoid(output.logits)
    probability_positive_class = probability[:, 1].item()  # Extract probability for class 1
    prediction = 1 if probability_positive_class >= 0.5 else 0
    return prediction

def classify_depression(input_text):
    prediction = detect_depression(input_text, model_distilbert, tokenizer_distilbert,)
    return prediction
   
   
    
   



@app.route('/')
def index():
    return render_template('index.html')

from flask import jsonify



@app.route('/recommendation', methods=['POST'])
def recommendation():
    user_input = request.form['user_input']
    
    # Classify if the text is depressive or not
    is_depressive = classify_depression(user_input)
    
    # Generate response based on the classification
    if is_depressive:
        response = "I am sorry to hear that. It's ok if it happened. Be Strong and always seek professional help in case of serious symptoms. play some mind  games for refreshment"

        show_dropdown = True
    else:
        response = "I am there to help you out. Whenever you feel depressed about anything, feel free to chat with me and refresh your mind by playing some games."

        show_dropdown = False

    return jsonify({'prediction': response, 'show_dropdown': show_dropdown})


if __name__ == '__main__':
    app.run(debug=True)




