from django.urls import path
from . import views

# 이 변수는 다른 앱과의 URL 이름 충돌을 방지합니다.
app_name = 'mypage' 

urlpatterns = [
    # 주소: core/ (아무것도 없으면)
    # 처리: views.index 함수 호출
    # 이름: 'index'로 이 주소를 참조할 수 있습니다.
    path('', views.mypage, name='mypage'), 
    path('update-profile/', views.update_profile, name='update_profile'),
    path('change-password/', views.change_password, name='change_password'),
    path('delete-account/', views.delete_account, name='delete_account'),
    # 예시: path('about/', views.about, name='about'),
]