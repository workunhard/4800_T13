from django.shortcuts import render
from django.http import Http404
from django.urls import reverse
from django.views.generic import TemplateView
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertTokenizer, pipeline
import torch.nn.functional as F
from .forms import ModelChoiceForm

# # Load RoBERTa-IteraTer model and tokenizer once when the script starts
# tokenizer = AutoTokenizer.from_pretrained("wanyu/IteraTeR-ROBERTA-Intention-Classifier")
# model = AutoModelForSequenceClassification.from_pretrained("wanyu/IteraTeR-ROBERTA-Intention-Classifier")
#
# # Load bert_edit_intent_classification
# comment_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
# bert_model = AutoModelForSequenceClassification.from_pretrained("citruschao/bert_edit_intent_classification1")

# Define your label mapping
# id2label = {0: "clarity", 1: "coherence", 2: "fluency", 3: "style", 4: "meaning changed"}
# id3label = {0: "clarity", 1: "coherence", 2: "fluency", 3: "meaning-changed", 4: "other", 5: "style"}

#Define explanations
explanation = {0: "Text is more formal, concise, readable and understandable",
               1: "Fixed grammatical errors in the text",
               2: "Text is more cohesive, logically linked and consistent as a whole",
               3: "Better conveys the writer’s writing preferences, including emotions, tone, voice, etc.",
               4: "Updated/added information to the text"}


def homePageView(request):
    post_request = False
    post_success = False
    if request.method == 'POST':
        post_request = True
        post_success = True
    input1 = ''  # Default value
    input2 = ''  # Default value
    input3 = ''  # Default value
    predictions_index = 0  # Default value
    predictions = ''
    bert_predictions = ''  # For the bert model
    form = ModelChoiceForm(request.POST or None)
    if form.is_valid():
        model_choice = form.cleaned_data.get('model_choice')
        comment_model_choice = form.cleaned_data.get('comment_model_choice')

        if model_choice == 'IteraTeR_ROBERTA':
            tokenizer = AutoTokenizer.from_pretrained("wanyu/IteraTeR-ROBERTA-Intention-Classifier")
            model = AutoModelForSequenceClassification.from_pretrained("wanyu/IteraTeR-ROBERTA-Intention-Classifier")
            base_label = {0: "clarity", 1: "coherence", 2: "fluency", 3: "style", 4: "meaning-changed"}

        elif model_choice == 'BERT_edit':
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
            model = AutoModelForSequenceClassification.from_pretrained("citruschao/bert_edit_intent_classification2")
            base_label = {0: "clarity", 1: "coherence", 2: "fluency", 3: "meaning-changed", 4: "other", 5: "style"}

        elif model_choice == 'bart_large_mnli':
            classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
            base_label = ["clarity", "coherence", "fluency", "style", "meaning-changed"]

        if comment_model_choice == 'IteraTeR_ROBERTA':
            comment_tokenizer = AutoTokenizer.from_pretrained("wanyu/IteraTeR-ROBERTA-Intention-Classifier")
            comment_model = AutoModelForSequenceClassification.from_pretrained("wanyu/IteraTeR-ROBERTA-Intention-Classifier")
            comment_label = {0: "clarity", 1: "coherence", 2: "fluency", 3: "style", 4: "meaning-changed"}

        elif comment_model_choice == 'BERT_edit':
            comment_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
            comment_model = AutoModelForSequenceClassification.from_pretrained("citruschao/bert_edit_intent_classification2")
            comment_label = {0: "clarity", 1: "coherence", 2: "fluency", 3: "meaning-changed", 4: "other", 5: "style"}

        elif comment_model_choice == 'bart_large_mnli':
            classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
            comment_label = ["clarity", "coherence", "fluency", "style", "change"]

        input1 = request.POST.get('original', '')
        input2 = request.POST.get('revised', '')
        input3 = request.POST.get('suggested_revision', '')
        if input1 == input2:
            return render(request, 'home.html', {'output': 'No revision detected', 'input1': input1, 'input2': input2, 'input3': input3, 'form': form})

        if model_choice == 'IteraTeR_ROBERTA' or model_choice == 'BERT_edit':
            inputs = tokenizer(input1, input2, return_tensors='pt', truncation=True, padding=True)
            outputs = model(**inputs)
            predictions_index = outputs.logits.argmax(-1).item()
            predictions = base_label[predictions_index]

            print(outputs.logits)
            print(model_choice + " prediction: " + str(predictions))

        elif model_choice == 'bart_large_mnli':
            outputs = classifier(input1 + " " + input2, base_label)

            print(outputs)

        if comment_model_choice == 'IteraTeR_ROBERTA' or comment_model_choice == 'BERT_edit':
            bert_inputs = comment_tokenizer(input3, return_tensors='pt', truncation=True, padding=True)
            bert_outputs = comment_model(**bert_inputs)
            probabilities = F.softmax(bert_outputs.logits, dim=-1)
            bert_predictions_index = probabilities.argmax(-1).item()
            bert_predictions = comment_label[bert_predictions_index]

            print("Probabilities: ", probabilities)
            print(comment_model_choice + " prediction: " + str(bert_predictions))

        elif comment_model_choice == 'bart_large_mnli':
            bert_outputs = classifier(input3, comment_label)
            bert_predictions_index = bert_outputs['scores'].index(
                max(bert_outputs['scores']))  # Getting index of max score
            bert_predictions = bert_outputs['labels'][bert_predictions_index]
            print(bert_outputs)
            print(comment_model_choice + " prediction: " + bert_predictions)

    if comment_model_choice == 'bart_large_mnli':
        if predictions == 'meaning-changed':
            predictions == 'change'

    outputs_match = predictions == bert_predictions

    print(outputs_match)

    return render(request, 'home.html',
                  {'form': form,
                   'output': 'Edit intention: ' + str(predictions),
                   'explanation': 'Explanation: ' + explanation[predictions_index],
                   'bert_output': 'Bert prediction: ' + str(bert_predictions),
                   'input1': input1,
                   'input2': input2,
                   'input3': input3,
                   'outputs_match': outputs_match,
                   'post_request': post_request,
                   'post_success': post_success})


def aboutPageView(request):
    return render(request, 'about.html')


def contactPageView(request):
    return render(request, 'about.html')
