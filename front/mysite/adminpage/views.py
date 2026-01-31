from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse
from django.views.decorators.http import require_POST

# common 앱의 모델들
from common.models import User, App, Analytics, AppGenre 

def dashboard(request):
    """
    관리자 대시보드 메인 뷰
    """
    # ---------------------------------------------------------
    # 1. 로그인 여부 및 관리자 체크 (세션 키 범용 검사)
    # ---------------------------------------------------------
    user_idx = request.session.get('user') or \
               request.session.get('u_idx') or \
               request.session.get('u_id') or \
               request.session.get('user_id')

    if not user_idx:
        return redirect('/') 
    
    try:
        current_user = User.objects.get(pk=user_idx)
    except User.DoesNotExist:
        request.session.flush()
        return redirect('/')

    if not current_user.u_admin: 
        return render(request, 'common/error.html', {'message': '관리자 권한이 없습니다.'})
        
    # ---------------------------------------------------------
    # 2. 대시보드 데이터 구성
    # ---------------------------------------------------------
    
    total_apps = App.objects.count()
    total_reports = Analytics.objects.count()
    total_users = User.objects.count()

    genres = AppGenre.objects.all()

    apps_with_reports = []
    
    all_apps = App.objects.all().select_related('ag_idx') 

    for app in all_apps:
        # [수정된 부분] v_idx(버전)를 거쳐서 a_idx(앱)를 조회합니다.
        # 앱 -> 버전 -> 분석보고서 구조이므로 v_idx__a_idx 로 접근해야 합니다.
        reports = Analytics.objects.filter(v_idx__a_idx=app).order_by('-an_idx')
        
        genre_name = app.ag_idx.ag_name if app.ag_idx else "기타"

        apps_with_reports.append({
            'app': app,
            'genre_name': genre_name,
            'reports': reports
        })

    all_users = User.objects.all().order_by('-u_idx')

    context = {
        'user': current_user,
        'u_admin_flag': 1,
        'total_apps_count': total_apps,
        'total_reports_count': total_reports,
        'total_users_count': total_users,
        'genres': genres,
        'apps_with_reports': apps_with_reports,
        'all_users': all_users,
    }

    return render(request, 'adminpage/adminpage.html', context)


# --- API (삭제 기능) ---

@require_POST
def delete_report(request, report_id):
    """ 보고서 삭제 API """
    try:
        user_idx = request.session.get('user') or request.session.get('u_idx') or request.session.get('u_id')
        if not user_idx:
            return JsonResponse({'success': False, 'message': '로그인이 필요합니다.'}, status=403)
        
        try:
            admin_user = User.objects.get(pk=user_idx)
        except User.DoesNotExist:
             return JsonResponse({'success': False, 'message': '유효하지 않은 유저입니다.'}, status=403)

        if not admin_user.u_admin:
             return JsonResponse({'success': False, 'message': '권한이 없습니다.'}, status=403)

        report = get_object_or_404(Analytics, pk=report_id) 
        report.delete()
        
        return JsonResponse({'success': True})
        
    except Exception as e:
        return JsonResponse({'success': False, 'message': str(e)})


@require_POST
def delete_user(request, user_id):
    """ 유저 강제 탈퇴 API """
    try:
        user_idx = request.session.get('user') or request.session.get('u_idx') or request.session.get('u_id')
        if not user_idx:
            return JsonResponse({'success': False, 'message': '로그인이 필요합니다.'}, status=403)
            
        try:
            admin_user = User.objects.get(pk=user_idx)
        except User.DoesNotExist:
             return JsonResponse({'success': False, 'message': '유효하지 않은 유저입니다.'}, status=403)

        if not admin_user.u_admin:
             return JsonResponse({'success': False, 'message': '권한이 없습니다.'}, status=403)

        target_user = get_object_or_404(User, pk=user_id)
        
        if target_user.u_admin:
             return JsonResponse({'success': False, 'message': '관리자 계정은 삭제할 수 없습니다.'})

        target_user.delete()
        
        return JsonResponse({'success': True})
        
    except Exception as e:
        return JsonResponse({'success': False, 'message': str(e)})