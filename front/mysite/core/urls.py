from django.urls import path
from . import views

# 이 변수는 다른 앱과의 URL 이름 충돌을 방지합니다.
app_name = 'core' 

urlpatterns = [
    # 주소: core/ (아무것도 없으면)
    # 처리: views.index 함수 호출
    # 이름: 'index'로 이 주소를 참조할 수 있습니다.
    path('', views.index, name='index'),
    path('check-id/', views.check_id, name='check_id'),  # 아이디 중복 체크
    path('signup/', views.signup, name='signup'),  # 회원가입
    path('login/', views.login_view, name='login'),  # 로그인
    path('logout/', views.logout_view, name='logout'),  # 이 줄 추가!
    
    # 예시: path('about/', views.about, name='about'),
]