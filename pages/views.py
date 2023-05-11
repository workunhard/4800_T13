from django.shortcuts import render
from django.http import Http404
from django.urls import reverse
from django.views.generic import TemplateView
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load your model and tokenizer once when the script starts
tokenizer = AutoTokenizer.from_pretrained("wanyu/IteraTeR-ROBERTA-Intention-Classifier")
model = AutoModelForSequenceClassification.from_pretrained("wanyu/IteraTeR-ROBERTA-Intention-Classifier")

# Define your label mapping
id2label = {0: "clarity", 1: "coherence", 2: "fluency", 3: "style", 4: "meaning-changed"}


def homePageView(request):
    output = ''
    input1 = ''  # Default value
    input2 = ''  # Default value
    predictions = ''
    if request.method == 'POST':
        input1 = request.POST.get('original', '')
        input2 = request.POST.get('revised', '')

        # Prepare your inputs for the model
        inputs = tokenizer(input1, input2, return_tensors='pt', truncation=True, padding=True)
        # Make the prediction
        outputs = model(**inputs)
        predictions_index = outputs.logits.argmax(-1).item()
        predictions = id2label[predictions_index]

        print("Single prediction: " + str(predictions))

    return render(request, 'home.html',
                  {'output': str(predictions), 'before_sent': input1, 'after_sent': input2})
