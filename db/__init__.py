try:
    import pymysql
    pymysql.install_as_MySQLdb()
    pymysql.version_info = (1, 4, 3, "final", 0)
except:
    print("Warning: pymysql is not installed, cannot access remote database")