from django.shortcuts import render
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertTokenizer, pipeline
import torch.nn.functional as F
from .forms import ModelChoiceForm


#Define explanations
explanation = {0: "Text is more formal, concise, readable and understandable",
               1: "Fixed grammatical errors in the text",
               2: "Text is more cohesive, logically linked and consistent as a whole",
               3: "Better conveys the writer’s writing preferences, including emotions, tone, voice, etc.",
               4: "Updated/added information to the text"}

# Define your models' descriptions
model_description = {
    "IteraTeR_ROBERTA": "IteraTeR-RoBERTa is a RoBERTa-based transformer model pre-trained on a large corpus of English text and fine-tuned for the intention classification task. It is designed to understand the text and predict its intention.",
    "BERT_edit": "BERT_edit is based on the BERT model, which is pre-trained on a large corpus of English text. BERT_edit has been fine-tuned for the task of edit intention prediction, allowing it to understand and predict the intention behind revisions in the text.",
    "bart_large_mnli": "bart_large_mnli is a model based on the BART architecture and is fine-tuned for the task of natural language inference (NLI). In this context, it is used to understand and classify the intention behind textual revisions."
}


def homePageView(request):
    post_request = False
    post_success = False
    model_choice = ''  # Added initialization
    comment_model_choice = ''  # Added initialization
    input1 = ''  # Default value
    input2 = ''  # Default value
    input3 = ''  # Default value
    predictions_index = 0  # Default value
    predictions = ''
    bert_predictions = ''  # For the bert model
    outputs_match = False  # Default value
    revise_probabilities = None
    probabilities = None
    logits = None
    bert_logits = None
    label1 = None
    label2 = None

    form = ModelChoiceForm(request.POST or None)

    if form.is_valid():
        post_request = True
        post_success = True

        model_choice = form.cleaned_data.get('model_choice')
        comment_model_choice = form.cleaned_data.get('comment_model_choice')

        if model_choice == 'IteraTeR_ROBERTA':
            tokenizer = AutoTokenizer.from_pretrained("wanyu/IteraTeR-ROBERTA-Intention-Classifier")
            model = AutoModelForSequenceClassification.from_pretrained("wanyu/IteraTeR-ROBERTA-Intention-Classifier", low_cpu_mem_usage=True)
            base_label = {0: "clarity", 1: "coherence", 2: "fluency", 3: "style", 4: "meaning-changed"}

        elif model_choice == 'BERT_edit':
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
            model = AutoModelForSequenceClassification.from_pretrained("citruschao/bert_edit_intent_classification2", low_cpu_mem_usage=True)
            base_label = {0: "clarity", 1: "coherence", 2: "fluency", 3: "meaning-changed", 4: "other", 5: "style"}

        elif model_choice == 'bart_large_mnli':
            classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", low_cpu_mem_usage=True)
            base_label = ["clarity", "coherence", "fluency", "style", "meaning-changed"]

        if comment_model_choice == 'IteraTeR_ROBERTA':
            comment_tokenizer = AutoTokenizer.from_pretrained("wanyu/IteraTeR-ROBERTA-Intention-Classifier")
            comment_model = AutoModelForSequenceClassification.from_pretrained(
                "wanyu/IteraTeR-ROBERTA-Intention-Classifier", low_cpu_mem_usage=True)
            comment_label = {0: "clarity", 1: "coherence", 2: "fluency", 3: "style", 4: "meaning-changed"}

        elif comment_model_choice == 'BERT_edit':
            comment_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
            comment_model = AutoModelForSequenceClassification.from_pretrained(
                "citruschao/bert_edit_intent_classification2", low_cpu_mem_usage=True)
            comment_label = {0: "clarity", 1: "coherence", 2: "fluency", 3: "meaning-changed", 4: "other", 5: "style"}

        elif comment_model_choice == 'bart_large_mnli':
            classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", low_cpu_mem_usage=True)
            comment_label = ["clarity", "coherence", "fluency", "style", "change"]

        input1 = request.POST.get('original', '')
        input2 = request.POST.get('revised', '')
        input3 = request.POST.get('suggested_revision', '')

        if input1 == input2:
            return render(request, 'home.html',
                          {'output': 'No revision detected', 'input1': input1, 'input2': input2, 'input3': input3,
                           'form': form})

        if model_choice == 'IteraTeR_ROBERTA' or model_choice == 'BERT_edit':
            inputs = tokenizer(input1, input2, return_tensors='pt', truncation=True, padding=True)
            outputs = model(**inputs)
            revise_probabilities = F.softmax(outputs.logits, dim=-1)
            predictions_index = outputs.logits.argmax(-1).item()
            predictions = base_label[predictions_index]

            logits = str(revise_probabilities)

            print(outputs.logits)
            print(model_choice + " prediction: " + str(predictions))

        elif model_choice == 'bart_large_mnli':
            outputs = classifier(input1 + " " + input2, base_label)
            predictions_index = outputs['scores'].index(max(outputs['scores']))  # Getting index of max score
            predictions = outputs['labels'][predictions_index]

            logits = str(predictions)

            print(outputs)

        if comment_model_choice == 'IteraTeR_ROBERTA' or comment_model_choice == 'BERT_edit':
            bert_inputs = comment_tokenizer(input3, return_tensors='pt', truncation=True, padding=True)
            bert_outputs = comment_model(**bert_inputs)
            probabilities = F.softmax(bert_outputs.logits, dim=-1)
            bert_predictions_index = probabilities.argmax(-1).item()
            bert_predictions = comment_label[bert_predictions_index]

            bert_logits = str(probabilities)

            print("Probabilities: ", probabilities)
            print(comment_model_choice + " prediction: " + str(bert_predictions))

        elif comment_model_choice == 'bart_large_mnli':
            bert_outputs = classifier(input3, comment_label)
            bert_predictions_index = bert_outputs['scores'].index(
                max(bert_outputs['scores']))  # Getting index of max score
            bert_predictions = bert_outputs['labels'][bert_predictions_index]
            print(bert_outputs)
            print(comment_model_choice + " prediction: " + bert_predictions)

            bert_logits = str(bert_predictions)

            print(outputs)

        if predictions == bert_predictions:
            outputs_match = True

        label1 = str(base_label)
        label2 = str(comment_label)

    context = {
        'form': form,
        'output': 'Predicted revision edit intent: ' + str(predictions),
        'explanation': 'Explanation: ' + explanation[predictions_index],
        'bert_output': 'Predicted comment edit intent: ' + str(bert_predictions),
        'input1': input1,
        'input2': input2,
        'input3': input3,
        'outputs_match': outputs_match,
        'post_request': post_request,
        'post_success': post_success,
        'revise_probabilities': revise_probabilities,
        'probabilities': probabilities,
        'model_choice': model_choice,
        'comment_model_choice': comment_model_choice,
        'logits': logits,
        'bert_logits': bert_logits,
        'label1': label1,
        'label2': label2
    }

    # Add model descriptions to the context if the models were selected
    if model_choice:
        context['model_description'] = model_description[model_choice]
    if comment_model_choice:
        context['comment_model_description'] = model_description[comment_model_choice]

    return render(request, 'home.html', context)


def aboutPageView(request):
    return render(request, 'about.html')


def contactPageView(request):
    return render(request, 'about.html')
