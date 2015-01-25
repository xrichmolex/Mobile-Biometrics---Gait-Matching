#!usr/bin/python
 
import sys, csv, os, errno

# Define Constants
#PATH = "Distribution/Pocket/"
#PATH = "Distribution/Belt/"
DIR = "Templates/"
DEBUG = True
FILES = 31
#FILES = 6


def find_file(name, path):
    for root, dirs, files in os.walk(path):
	if name in files:
	    return True
	else:
	    return False


def removeFile(filename):
    try:
        os.remove(filename)
    except OSError as e:
        if e.errno != errno.ENOENT: 
            raise 

def main(argv):
 
    type = argv[0]

    if type == 'belt':
	PATH = "Distribution/Belt/"
    else:
	PATH = "Distribution/Pocket/"


    template = type+".csv"
    tem = DIR+template

    for x in range(FILES):
	file = type + str(x) + ".csv"

	#Create Template
	call = "python method.py "+PATH+file+" "+template
	print "Template: "+call
	os.system(call)
	
	for i in range(x+1,FILES):
	    file = type + str(i) + ".csv"

	    call = "python method.py "+PATH+file+" "+template
	    print "Probe: "+call
	    os.system(call)
	    

	print "removing file..."
	removeFile(tem)
	print "file removed"

 
if __name__ == "__main__":
    main(sys.argv[1:])
