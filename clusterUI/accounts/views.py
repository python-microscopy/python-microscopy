from django.shortcuts import render

# Create your views here.
from django.contrib.auth.forms import PasswordResetForm
from django.shortcuts import redirect
from django.views.generic import CreateView

from .forms import RegistrationForm
from .models import User

class RegistrationView(CreateView):
    form_class = RegistrationForm
    model = User

    def form_valid(self, form):
        obj = form.save(commit=False)
        #obj.set_password(User.objects.make_random_password())
        #obj.save()

        user = User.objects.create_user(obj.email, User.objects.make_random_password())

        # This form only requires the "email" field, so will validate.
        reset_form = PasswordResetForm(self.request.POST)
        reset_form.is_valid()  # Must trigger validation
        # Copied from django/contrib/auth/views.py : password_reset
        opts = {
            'use_https': self.request.is_secure(),
            'email_template_name': 'registration/verification.html',
            'subject_template_name': 'registration/verification_subject.txt',
            'request': self.request,
            # 'html_email_template_name': provide an HTML content template if you desire.
        }
        # This form sends the email on save()
        reset_form.save(**opts)

        return redirect('accounts:register-done')