import threading
import subprocess
def run_paraelly(command):
    script_name=command[1]
    result =subprocess.run(command,capture_output=True,text=True)
    if result.returncode==0:
        print("sucess")
    else:
        print(f"Error running {script_name}:\n{result.stderr}")
script1_command=["python","sensor.py"]
script2_command=["python","main_api.py","-m","last_tiny.blob","-c","last_tiny.json","-r","demo.avi"]
#create a thread
script1_thread1=threading.Thread(target=run_paraelly,args=(script1_command,))
script2_thread2=threading.Thread(target=run_paraelly,args=(script2_command,))
script1_thread1.start()
print("thread 1 is start")
script2_thread2.start()
print("thread2 is start")
#merge them together
script1_thread1.join()
script2_thread2.join()



