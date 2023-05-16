from django.shortcuts import render
from django.http import Http404
from django.urls import reverse
from django.views.generic import TemplateView
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertTokenizer
import torch.nn.functional as F

# Load RoBERTa-IteraTer model and tokenizer once when the script starts
tokenizer = AutoTokenizer.from_pretrained("wanyu/IteraTeR-ROBERTA-Intention-Classifier")
model = AutoModelForSequenceClassification.from_pretrained("wanyu/IteraTeR-ROBERTA-Intention-Classifier")

# Load bert_edit_intent_classification
comment_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
bert_model = AutoModelForSequenceClassification.from_pretrained("citruschao/bert_edit_intent_classification1")

# Define your label mapping
id2label = {0: "clarity", 1: "coherence", 2: "fluency", 3: "style", 4: "meaning changed"}
id3label = {0: "clarity", 1: "coherence", 2: "fluency", 3: "meaning-changed", 4: "other", 5: "style"}

#Define explanations
explanations = {0: "Text is more formal, concise, readable and understandable",
               1: "Fixed grammatical errors in the text",
               2: "Text is more cohesive, logically linked and consistent as a whole",
               3: "Better conveys the writer’s writing preferences, including emotions, tone, voice, etc.",
               4: "Updated/added information to the text"}


def homePageView(request):
    input1 = ''  # Default value
    input2 = ''  # Default value
    input3 = ''  # Default value
    predictions_index = 0
    predictions = ''
    explanation = ''
    bert_predictions = ''  # For the bert model
    if request.method == 'POST':
        input1 = request.POST.get('original', '')
        input2 = request.POST.get('revised', '')
        input3 = request.POST.get('suggested_revision', '')
        if input1 == input2:
            return render(request, 'home.html',
                          {'output': 'No revision detected', 'input1': input1, 'input2': input2, 'input3': input3})

        # Prepare your inputs for the IteraTeR-ROBERTA-Intention-Classifier model
        inputs = tokenizer(input1, input2, return_tensors='pt', truncation=True, padding=True)

        # Make the prediction
        outputs = model(**inputs)
        predictions_index = outputs.logits.argmax(-1).item()
        predictions = id2label[predictions_index]
        explanation = explanations[predictions_index]

        print(outputs.logits)
        print("Single prediction: " + str(predictions))

        # Now, process input3 with the bert_edit_intent_classification1 model
        bert_inputs = comment_tokenizer("[CLS]" + input3 + "[SEP]", return_tensors='pt', truncation=True, padding=True)
        bert_outputs = bert_model(**bert_inputs)

        # Apply softmax to logits
        probabilities = F.softmax(bert_outputs.logits, dim=-1)

        bert_predictions_index = probabilities.argmax(-1).item()
        bert_predictions = id3label[bert_predictions_index]

        # Print probabilities
        print("Probabilities: ", probabilities)
        print("Bert prediction: " + str(bert_predictions))

    return render(request, 'home.html',
                  {'output': 'Edit intention: ' + str(predictions),
                   'explanation': 'Explanation: ' + str(explanation),
                   'bert_output': 'Bert prediction: ' + str(bert_predictions),
                   'input1': input1,
                   'input2': input2,
                   'input3': input3})


def aboutPageView(request):
    return render(request, 'about.html')


def contactPageView(request):
    return render(request, 'about.html')
                   'input3': input3})
