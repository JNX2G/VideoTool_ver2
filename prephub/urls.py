from django.urls import path
from . import views

app_name = 'prephub'

urlpatterns = [
    path('', views.MethodListView.as_view(), name='method_list'),
    path('add/', views.MethodCreateView.as_view(), name='method_add'),
    path('<int:pk>/', views.MethodDetailView.as_view(), name='method_detail'),
    path('<int:pk>/edit/', views.MethodUpdateView.as_view(), name='method_edit'),
    path('<int:pk>/delete/', views.MethodDeleteView.as_view(), name='method_delete'),

    # 일괄 제어
    path('bulk-toggle/', views.BulkToggleActiveView.as_view(), name='bulk_toggle'),
    path('<int:pk>/toggle/', views.ToggleActiveView.as_view(), name='toggle_active'),
]