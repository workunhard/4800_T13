from django import forms


class ModelChoiceForm(forms.Form):
    MODEL_CHOICES = [
        ('IteraTeR_ROBERTA', 'IteraTeR ROBERTA Intention Classifier'),
        ('BERT_edit', 'BERT Edit Intent Classification1'),
        ('bart_large_mnli', 'bart_large_mnli Zero Shot Text Classification'),
    ]

    model_choice = forms.ChoiceField(choices=MODEL_CHOICES, required=True)
    comment_model_choice = forms.ChoiceField(choices=MODEL_CHOICES, required=True)