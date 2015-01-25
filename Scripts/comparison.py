#!usr/bin/python
 
import sys, csv, os, errno

# Define Constants
PATH = "Distribution/Pocket/pocket"
#PATH = "Distribution/Belt/belt"
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

def main(argv):
 
    name = argv[0]

    template = name+".csv"
    tem = DIR+template

    #Create Template (if necessary)
    if not find_file(template,DIR):
	call = "python method.py "+file+" "+template
	print "Template: "+call
	os.system(call)

    for x in range(FILES):
	file = PATH + str(x) + ".csv"

	call = "python method.py "+file+" "+template
	print "Probe: "+call
	os.system(call)
	    
 
if __name__ == "__main__":
    main(sys.argv[1:])
