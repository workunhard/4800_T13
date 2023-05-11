from django.shortcuts import render
from django.shortcuts import render, HttpResponseRedirect
from django.http import Http404
from django.urls import reverse
from django.views.generic import TemplateView

def homePageView(request):
    output = ''
    input1 = ''  # Default value
    input2 = ''  # Default value
    if request.method == 'POST':
        input1 = request.POST.get('original', '')
        input2 = request.POST.get('revised', '')
        output = input1 + ' ' + input2
    return render(request, 'home.html', {'output': output, 'input1': input1, 'input2': input2})
