from goatools.obo_parser import GODag

# 在模块加载时只执行一次，加载 GO DAG
go_obo_path = "go_basic.obo"
go_dag = GODag(go_obo_path)

def get_go_dag():
    return go_dag
