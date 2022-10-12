import os

def remove_exist_file(filename):
    if (os.path.exists(filename) and os.path.isfile(filename)):
        os.remove(filename)
        print("remove exist file: ", filename)