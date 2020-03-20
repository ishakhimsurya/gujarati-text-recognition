from django.shortcuts import render, redirect 
from django.http import HttpResponse 
from .forms import CharForm

# Create your views here.

def char_image_view(request): 

	if request.method == 'POST': 
		form = CharForm(request.POST, request.FILES) 

		if form.is_valid(): 
			form.save() 
			return redirect('success') 
	else: 
		form = CharForm() 
	return render(request, 'image.html', {'form' : form}) 


def success(request): 
	return HttpResponse('successfully uploaded') 



# class ProfileView(View):
#     model = Profile

#     def get(self, request, *args, **kwargs):
#         return render(request, 'image.html')

#     def post(self, request, *args, **kwargs):
#         pic = self.model()
#         try:
#             pic.picture = request.FILES.get("picture")
#             pic.save()
#             filename = pic.picture.name
#             text = image_to_string(Image.open(settings.MEDIA_ROOT + '/' + filename))
#             # print(text)
#         except Exception:
#             pass
#         return render(request, 'saved.html', locals())