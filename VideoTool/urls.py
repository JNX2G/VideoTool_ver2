"""
URL configuration for VideoTool project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from django.views.generic import RedirectView

urlpatterns = [
    path("admin/", admin.site.urls),

    # 앱 경로
    path("contents/", include("contents.urls")),
    path("prephub/", include("prephub.urls")),
    path("preprocess/", include("preprocess.urls")),
    path("modelhub/", include("modelhub.urls")),
    path("vision_engine/", include("vision_engine.urls")), 

    # 이미지 비교
    path("image_compare/", include("image_compare.urls")), 

    path('favicon.ico', RedirectView.as_view(url='/static/favicon.ico', permanent=True)),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static('/models/', document_root=settings.MODELS_ROOT)
    urlpatterns += static('/results/', document_root=settings.RESULTS_ROOT)
