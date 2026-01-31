from django.urls import path
from . import views

# 이 변수는 다른 앱과의 URL 이름 충돌을 방지합니다.
app_name = 'main' 

urlpatterns = [
    # 주소: core/ (아무것도 없으면)
    # 처리: views.index 함수 호출
    # 이름: 'index'로 이 주소를 참조할 수 있습니다.
    path('', views.main, name='main'),
    path('report/<int:report_id>/', views.get_report, name='get_report'),
    path('report/<int:report_id>/download/', views.download_report_pdf, name='download_report_pdf'),
    path('api/chat/', views.chat_api, name='chat_api'),
    path('api/report/<int:report_id>/toggle-save/', views.toggle_save_report, name='toggle_save_report'),
    
    # 예시: path('about/', views.about, name='about'),
]