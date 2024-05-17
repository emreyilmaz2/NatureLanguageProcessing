from django.urls import path
from .views import register, login_view, logout_view, user_profile, search_articles
from rest_framework_simplejwt.views import (
    TokenRefreshView,
)

urlpatterns = [
    path('register/', register, name='register'),
    path('login/', login_view, name='login'),
    path('logout/', logout_view, name='logout'),
    path('profile/', user_profile, name='profile'),
    path('profile/update/', user_profile, name='profile'),
    path('search-article/', search_articles, name='search_articles'),
    path('token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
]