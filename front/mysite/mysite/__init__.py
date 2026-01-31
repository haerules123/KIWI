import pymysql

# Django의 MySQL 버전 체크를 통과하기 위한 속임수 (Bypass)
pymysql.version_info = (2, 2, 7, "final", 0) 

# pymysql을 MySQLdb처럼 작동하게 만듦
pymysql.install_as_MySQLdb()