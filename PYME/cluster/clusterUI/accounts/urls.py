from django.urls import re_path

from .views import RegistrationView
from django.contrib.auth import views

urlpatterns = [
    re_path(r'^register/$', RegistrationView.as_view(), name='register'),
    re_path(r'^register/done/$', views.PasswordResetDoneView.as_view(), {
        'template_name': 'registration/initial_done.html',
    }, name='register-done'),

    re_path(r'^register/password/(?P<uidb64>[0-9A-Za-z_\-]+)/(?P<token>[0-9A-Za-z]{1,13}-[0-9A-Za-z]{1,20})/$', views.PasswordResetConfirmView.as_view(), {
        'template_name': 'registration/initial_confirm.html',
        'post_reset_redirect': 'accounts:register-complete',
    }, name='register-confirm'),
    re_path(r'^register/complete/$', views.PasswordResetCompleteView.as_view(), {
        'template_name': 'registration/initial_complete.html',
    }, name='register-complete'),
]