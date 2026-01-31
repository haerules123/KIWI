from django.urls import path
from . import views

app_name = 'adminpage'  # 네임스페이스 지정 (선택사항이지만 권장)

urlpatterns = [
    # 대시보드 메인 페이지
    path('', views.dashboard, name='dashboard'),

    # 삭제 API 엔드포인트
    # HTML JS에 적힌 경로: /adminpage/api/delete-report/...
    path('api/delete-report/<int:report_id>/', views.delete_report, name='delete_report'),
    path('api/delete-user/<int:user_id>/', views.delete_user, name='delete_user'),
]