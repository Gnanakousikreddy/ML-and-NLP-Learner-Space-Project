from django.shortcuts import render
from .transformer_model import generator
# Create your views here.


def generate(request) :
    generated_text = ""
    if request.method == "POST" :
        prompt = request.POST.get("prompt", "")
        if prompt :
            generated_text = generator.generate_text(prompt)
    context = {"generated_text" : generated_text}
    return render(request, "textgen/generate.html", context)



