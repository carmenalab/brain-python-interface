try:
    import pymysql
    pymysql.install_as_MySQLdb()
except:
    print("Warning: pymysql is not installed, cannot access remote database")