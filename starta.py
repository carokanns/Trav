import os

os.chdir("spel")
answer = input("Do you want to remove v75.log? (y/n)")

if answer.lower() == 'y':
    try:
        os.remove("v75.log")
        print("v75.log removed.")
    except:
        print("v75.log not found.")
else:
    print("v75.log not removed.")
    
os.system("streamlit run start.py")
