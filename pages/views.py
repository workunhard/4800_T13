from django.shortcuts import render
from django.shortcuts import render, HttpResponseRedirect
from django.http import Http404
from django.urls import reverse
from django.views.generic import TemplateView

def homePageView(request):
    output = ''
    if request.method == 'POST':
        input1 = request.POST['original']
        input2 = request.POST['revised']
        output = input1 + ' ' + input2
    return render(request, 'home.html', {'output': output, 'input1': input1, 'input2': input2})
