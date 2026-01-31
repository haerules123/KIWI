from django.shortcuts import redirect
from functools import wraps

def login_required(view_func):
    """
    로그인이 필요한 페이지에 사용하는 데코레이터
    세션에 user_id가 없으면 index 페이지로 리다이렉트
    """
    @wraps(view_func)
    def wrapper(request, *args, **kwargs):
        # 세션에 user_id가 있는지 확인
        if 'user_id' not in request.session:
            # 로그인 안 되어있으면 index로 리다이렉트
            return redirect('core:index')
        
        # 로그인 되어있으면 원래 뷰 함수 실행
        return view_func(request, *args, **kwargs)
    
    return wrapper