# forms.py 
from django import forms 
from .models import *

class CharForm(forms.ModelForm): 

	class Meta: 
		model = Profile
		fields = ['picture'] 
