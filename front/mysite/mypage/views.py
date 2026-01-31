from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.contrib.auth.hashers import check_password
from django.db import connection
import json
from common.models import User, Saved, Analytics, Version, App
from django.db import connection

@require_http_methods(["GET"])
def mypage(request):
    """마이페이지 메인"""
    if 'user_id' not in request.session:
        return redirect('core:index')
    
    u_idx = request.session['user_id']
    cursor = connection.cursor()
    
    # 1. 유저 정보 조회
    cursor.execute("""
        SELECT u_name, u_email, u_id
        FROM user 
        WHERE u_idx = %s
    """, [u_idx])
    user_data = cursor.fetchone()
    
    if not user_data:
        cursor.close()
        return redirect('core:index')
    
    # 2. 앱 리스트 조회
    cursor.execute("""
        SELECT a.a_idx, a.a_name, a.a_developer, a.a_icon
        FROM user_app_list ual
        JOIN app a ON ual.a_idx = a.a_idx
        WHERE ual.u_idx = %s
    """, [u_idx])
    user_apps = cursor.fetchall()
    
    user_company = user_apps[0][2] if user_apps else "소속 없음"
    user_app_icon = user_apps[0][3] if user_apps else None
    
    apps_list = [
        {'a_idx': app[0], 'a_name': app[1], 'a_developer': app[2]}
        for app in user_apps
    ]

    # 저장된 보고서 리스트 조회 시작
    cursor.execute("""
        SELECT 
            an.an_idx,
            a.a_name,
            v.v_version,
            CONCAT(YEAR(MIN(r.r_date)), ' Q', QUARTER(MIN(r.r_date))) as report_quarter
        FROM saved s
        JOIN analytics an ON s.an_idx = an.an_idx
        JOIN version v ON an.v_idx = v.v_idx
        JOIN app a ON v.a_idx = a.a_idx
        LEFT JOIN review r ON r.v_idx = v.v_idx
        WHERE s.u_idx = %s
        GROUP BY an.an_idx, a.a_name, v.v_version, s.s_idx
        ORDER BY s.s_idx DESC
    """, [u_idx])
    
    saved_rows = cursor.fetchall()
    
    saved_reports = []
    for row in saved_rows:
        saved_reports.append({
            'an_idx': row[0],
            'app_name': row[1],
            'version': row[2],
            'quarter': row[3] if row[3] else "분석 기간 없음"
        })
    # ==========================================
    # [추가된 부분] 끝
    # ==========================================
    
    context = {
        'user_name': user_data[0],
        'user_email': user_data[1],
        'user_company': user_company,
        'user_apps': apps_list,
        'user_app_icon': user_app_icon,
        'saved_reports': saved_reports,
    }
    
    cursor.close()
    return render(request, 'mypage/mypage.html', context)


@require_http_methods(["POST"])
def update_profile(request):
    """프로필 수정"""
    if 'user_id' not in request.session:
        return JsonResponse({'success': False, 'message': '로그인이 필요합니다.'})
    
    try:
        data = json.loads(request.body)
        new_name = data.get('name')
        new_email = data.get('email')
        u_idx = request.session['user_id']  # u_idx
        
        cursor = connection.cursor()
        
        # 이메일 중복 체크
        cursor.execute("""
            SELECT u_idx FROM user 
            WHERE u_email = %s AND u_idx != %s
        """, [new_email, u_idx])
        
        if cursor.fetchone():
            cursor.close()
            return JsonResponse({'success': False, 'message': '이미 사용 중인 이메일입니다.'})
        
        # 프로필 업데이트
        cursor.execute("""
            UPDATE user 
            SET u_name = %s, u_email = %s 
            WHERE u_idx = %s
        """, [new_name, new_email, u_idx])
        
        connection.commit()
        cursor.close()
        
        request.session['user_name'] = new_name
        
        return JsonResponse({'success': True, 'message': '프로필이 수정되었습니다.'})
        
    except Exception as e:
        return JsonResponse({'success': False, 'message': str(e)})

@require_http_methods(["POST"])
def change_password(request):
    """비밀번호 변경"""
    if 'user_id' not in request.session:
        return JsonResponse({'success': False, 'message': '로그인이 필요합니다.'})
    
    try:
        data = json.loads(request.body)
        current_pw = data.get('current_password')
        new_pw = data.get('new_password')
        u_idx = request.session['user_id']
        
        cursor = connection.cursor()
        
        # 현재 비밀번호 가져오기
        cursor.execute("""
            SELECT u_pw FROM user 
            WHERE u_idx = %s
        """, [u_idx])
        
        result = cursor.fetchone()
        if not result:
            cursor.close()
            return JsonResponse({'success': False, 'message': '사용자를 찾을 수 없습니다.'})
        
        stored_password = result[0]
        
        # Django의 check_password로 비밀번호 확인
        if not check_password(current_pw, stored_password):
            cursor.close()
            return JsonResponse({'success': False, 'message': '현재 비밀번호가 일치하지 않습니다.'})
        
        # 새 비밀번호로 업데이트 (Django 해싱 사용)
        from django.contrib.auth.hashers import make_password
        hashed_new = make_password(new_pw)
        
        cursor.execute("""
            UPDATE user 
            SET u_pw = %s 
            WHERE u_idx = %s
        """, [hashed_new, u_idx])
        
        connection.commit()
        cursor.close()
        
        return JsonResponse({'success': True, 'message': '비밀번호가 변경되었습니다.'})
        
    except Exception as e:
        return JsonResponse({'success': False, 'message': str(e)})


@require_http_methods(["POST"])
def delete_account(request):
    """회원 탈퇴"""
    if 'user_id' not in request.session:
        return JsonResponse({'success': False, 'message': '로그인이 필요합니다.'})
    
    try:
        data = json.loads(request.body)
        input_password = data.get('password')
        
        if not input_password:
            return JsonResponse({'success': False, 'message': '비밀번호를 입력해주세요.'})
        
        u_idx = request.session['user_id']
        cursor = connection.cursor()
        
        # 현재 비밀번호 가져오기
        cursor.execute("""
            SELECT u_pw FROM user 
            WHERE u_idx = %s
        """, [u_idx])
        
        result = cursor.fetchone()
        if not result:
            cursor.close()
            return JsonResponse({'success': False, 'message': '사용자를 찾을 수 없습니다.'})
        
        stored_password = result[0]
        
        # Django의 check_password로 비밀번호 확인
        if not check_password(input_password, stored_password):
            cursor.close()
            return JsonResponse({'success': False, 'message': '비밀번호가 일치하지 않습니다.'})
        
        # 비밀번호 일치 → 탈퇴 진행
        # user_app_list 먼저 삭제
        cursor.execute("""
            DELETE FROM user_app_list 
            WHERE u_idx = %s
        """, [u_idx])
        
        # user 삭제
        cursor.execute("DELETE FROM user WHERE u_idx = %s", [u_idx])
        
        connection.commit()
        cursor.close()
        
        # 세션 삭제
        request.session.flush()
        
        return JsonResponse({'success': True, 'message': '계정이 삭제되었습니다.'})
        
    except Exception as e:
        return JsonResponse({'success': False, 'message': str(e)})