#!usr/bin/python
 
import sys, csv, os
from math import pow,sqrt,sin,cos,acos,pi
import math

# Define Constants
PI = pi
DIR = "Templates/"
STANDARD_LENGTH = 100
INTERVAL = 0.05
THRESHOLD = STANDARD_LENGTH-20
DEBUG = True
TRANSFORM = False
DISTRIBUTION = False

# Global Variables
average = 0.0
time = []
x_acc = []
y_acc = []
z_acc = []
acc_x = []
acc_y = []
acc_z = []
grav_x = []
grav_y = []
grav_z = []
peaks = []
markers = []
markers_2 = []
gallary_x = []
gallary_y = []
gallary_z = []
Rx = [[0 for x in range(3)] for x in range(3)]
Ry = [[0 for x in range(3)] for x in range(3)]
Rz = [[0 for x in range(3)] for x in range(3)]


def parse_data(input):
    global average
    global acc_x
    global acc_y
    global acc_z

    print "\n==== READING DATA FILE ===="

    ## read data ## 
    with open(input, "rU") as infile:
	reader = csv.reader(infile)
	count = 0
	sum = 0.0
	for line in reader:
	    if count == 0:
		count+=1
		continue 
	    time.append(line[0]);
	    x_acc.append(float(line[1]));
	    y_acc.append(float(line[2]));
	    z_acc.append(float(line[3]));
	    
	    sum += float(line[2])
	    count+=1
	    
	    grav_x.append(float(line[4]));
	    grav_y.append(float(line[5]));
	    grav_z.append(float(line[6]));

	average = sum / count
	print "Number of Sample Points: "+str(count-1)
	print "Average y_acceleration: "+str(average)

    if not TRANSFORM:
	print "NO DATA TRANSFORMATION"
	acc_x = x_acc
	acc_y = y_acc
	acc_z = z_acc

    print "==== COMPLETE ====\n"

 
def build_rotation_matrix(angx, angy, angz):

    # Global Rotation matricies
    global Rx
    global Ry
    global Rz

    # Reset rotation matrices to zeroes
    Rx = [[0 for x in range(3)] for x in range(3)]
    Ry = [[0 for x in range(3)] for x in range(3)]
    Rz = [[0 for x in range(3)] for x in range(3)]

    # Default values
    Rx[0][0] = 1
    Ry[1][1] = 1
    Rz[2][2] = 1

    # Build x-axis rotation matrix
    Rx[1][1] = cos(angx)
    Rx[1][2] = -sin(angx)
    Rx[2][1] = sin(angx)
    Rx[2][2] = cos(angx)

    # Build y-axis rotation matrix
    Ry[0][0] = cos(angy)
    Ry[0][2] = sin(angy)
    Ry[2][0] = -sin(angy)
    Ry[2][2] = cos(angy)

    # Build z-axis rotation matrix
    Rz[0][0] = cos(angz)
    Rz[0][1] = -sin(angz)
    Rz[1][0] = sin(angz)
    Rz[1][1] = cos(angz)

    #print "Rotation Matrix built"

def matrix_multiplication(X, Y):

    result = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*Y)] for X_row in X]
    return result

def modify_data():

    print "==== BEGIN DATA TRANSFORMATION ===="

    #global acc_x
    #global acc_y
    #global acc_z
    #global x_acc
    #global y_acc
    #global z_acc
   
    if DEBUG:
	print "Writing current data to:   modified.csv\n"

	out = open("modified.csv", 'w')

	header = "time,acc_x,acc_y,acc_z,gravity_x,gravity_y,gravity_z\n"
	out.write(header)

    for index in range(len(x_acc)):
	# Create 3x1 matrix of acceleration and gravity values
	vertices = [[x_acc[index]], [y_acc[index]], [z_acc[index]]]
	gravity = [[grav_x[index]], [grav_y[index]], [grav_z[index]]]

	#print "before rotation"
	#print vertices
	#print gravity
	
	if DEBUG:
	    output = time[index] + ','

	# ROTATE ON X-AXIS
	#grav_xy = sqrt(pow(grav_x[index],2) + pow(grav_y[index],2))
	#grav_xz = sqrt(pow(grav_x[index],2) + pow(grav_z[index],2))
	grav_yz = sqrt(pow(grav_y[index],2) + pow(grav_z[index],2))

	angx = float(acos(grav_y[index] / grav_yz))
	#angy = float(acos(grav_y[index] / grav_xz))
	#angz = float(acos(grav_y[index] / grav_xy))

	# Determine quadrant vector is in, modify angle if necessary
	if grav_y[index] > 0: 
	    if grav_z[index] > 0: 
		angx = PI - angx
	    else:
		angx = PI + angx
	else:
	    if grav_z[index] > 0: 
		angx = (2*PI) - angx


	# Perform Rotation
	#print "Rotate on X-axis by "+str(angx)+" radians"
	build_rotation_matrix(angx, 0, 0)	#Rotation matrix for x only
	acc1 = matrix_multiplication(Rx, vertices)
	grav1 = matrix_multiplication(Rx, gravity)

	# ROTATE ON Z-AXIS
	grav_xy = sqrt(pow(grav1[0][0],2) + pow(grav1[1][0],2))
	angz = float(acos(grav1[1][0] / grav_xy))

	if grav1[0][0] < 0: 
	    angz = (2*PI) - angz

	# Rotate
	#print "Rotate on Z-axis by "+str(angx)+" radians"
	build_rotation_matrix(0, 0, angz)	#Rotation matrix for z only
	acc2 = matrix_multiplication(Rz, acc1)
	grav2 = matrix_multiplication(Rz, grav1)

	#print "after x rotation"
	#print acc1
	#print grav1

	acc_x.append(float(acc2[0][0]))
	acc_y.append(float(acc2[1][0]))
	acc_z.append(float(acc2[2][0]))

	if DEBUG:
	    output += str(acc2[0][0]) + ',' + str(acc2[1][0]) + ',' + str(acc2[2][0]) + ',' 
	    output += str(grav2[0][0]) + ',' + str(grav2[1][0]) + ',' + str(grav2[2][0]) + ',' 
	    output += '\n'
	    out.write(output)

    print "==== COMPLETE ====\n"

def local_max():
    # CALCULATE THE LOCAL MAXIMUMS IN THE ACC_Y VALUES
    # FILTERING CODE BASED ON 40-point window
    #     (point must be maximum of 20 points on either side)

    global peaks

    entries = len(acc_y)
    count = 0

    for x in range (20, entries-20):

	values = acc_y[x-20:x+21]
	curr = acc_y[x]
	max_y = max(values)

	if max_y == curr and curr > average+2:
	    peaks.append(curr)
	    count+=1
	else:
	    peaks.append(float(-50))


    # Account for values skipped at begining/end of list
    bookend = [-50] * 20
    peaks = bookend + peaks + bookend

    print "Local Max Points: "+str(count)

    if DEBUG:
	output_local_max()

def output_local_max():
    #global time
    #global acc_x
    #global acc_y
    #global acc_z
    #global peaks

    print "Writing current data to:   local_max.csv\n"

    out = open("local_max.csv", 'w')

    header = "time,acc_x,acc_y,acc_z,local max\n"
    out.write(header)

    for index in range(len(acc_x)):
	output = time[index] + ',' + str(acc_x[index]) + ',' + str(acc_y[index]) + ',' + str(acc_z[index]) + ',' + str(peaks[index]) +'\n'
	#print output
	out.write(output)

    out.close()

# DIVIDE DATA INTO GAIT CYCLES
def identify_cycles():
    counter = 0
    begin = 0
    mid = 0
    end = 0
    
    for index in range(len(peaks)):
	if peaks[index] == -50:
	    continue
	else:
	    counter += 1

	    if counter == 1:
		begin = index
		continue

	    if counter == 2:
		mid = index
		continue

	    if counter == 3:
		end = index
		     
		points = [begin,mid,end]	
		#print points
		markers.append(points)

		counter = 1
		begin = end	

    print "Cycles Detected: "+str(len(markers))
    return

# DIVIDE DATA INTO GAIT CYCLES
def identify_dual_cycles():
    counter_1 = 0
    begin_1 = 0
    mid_1 = 0
    end_1 = 0

    counter_2 = -1
    begin_2 = 0
    mid_2 = 0
    end_2 = 0
    
    for index in range(len(peaks)):
	if peaks[index] == -50:
	    continue
	else:
	    #print index

	    counter_1 += 1
	    counter_2 += 1

	    if counter_1 == 1:
		begin_1 = index
		continue

	    if counter_2 == 3:
		end_2 = index
		
		points = [begin_2,mid_2,end_2]	
		#print points
		markers_2.append(points)

		counter_2 = 1
		begin_2 = end_2

	    if counter_1 == 2:
		mid_1 = index
		begin_2 = index
		continue

	    if counter_1 == 3:
		end_1 = index
		mid_2 = index
		     
		points = [begin_1,mid_1,end_1]	
		#print points
		markers.append(points)

		counter_1 = 1
		begin_1 = end_1	

    print "'a' Cycles Detected: "+str(len(markers))
    print "'b' Cycles Detected: "+str(len(markers_2))+"\n"
    return

def normalize(marks):
    # NORMALIZE STEPS
    # AMPLITUDE - DIVIDE BY MAX Y-VALUE
    # LENGTH - HAVE COMMON LENGTH (100 data points) AND FIT TO THAT LENGTH USING LINEAR INTERPOLATION
    
    print "Normalizing cycles..."

    height = 0
    steps = []

    for index in range(len(marks)):

	#new_x = [None] * STANDARD_LENGTH
	#new_y = [None] * STANDARD_LENGTH
	#new_z = [None] * STANDARD_LENGTH
        normal = []

	begin = marks[index][0]
	mid = marks[index][1]
	end = marks[index][2]

	#get points in cycle
	x = acc_x[begin:end+1]
	y = acc_y[begin:end+1]
	z = acc_z[begin:end+1]

	#adjust markers
	end = end - begin
	mid = mid - begin
	begin = 0

	#normalize height
	height = max(abs(y[begin]),abs(y[mid]),abs(y[end]))
	for i in range(len(y)):
	    x[i] = x[i] / height
	    y[i] = y[i] / height
	    z[i] = z[i] / height

	#normalize length
	length = float(end) / (STANDARD_LENGTH-1)
	interval = 0
	for i in range(0,STANDARD_LENGTH):
	    if i == 0:
		points = [x[begin],y[begin],z[begin]]	
		normal.append(points)
		#new_x[i] = x[begin]
		#new_y[i] = y[begin]
		#new_z[i] = z[begin]

	    elif i == STANDARD_LENGTH-1:
		points = [x[end],y[end],z[end]]	
		normal.append(points)
		#new_x[i] = x[end]
		#new_y[i] = y[end]
		#new_z[i] = z[end]

	    else:
		interval = length*i
		#print interval	
		prev = int(interval)
		next = prev+1
		#print prev
		#print next
		gap = interval - prev
		#print gap
		#print y[next]
		#print y[prev]
		diff_x = x[next] - x[prev]
		diff_y = y[next] - y[prev]
		diff_z = z[next] - z[prev]
		#print diff_y

		new_x = x[prev] + (gap * diff_x)
		new_y = y[prev] + (gap * diff_y)
		new_z = z[prev] + (gap * diff_z)

		points = [new_x,new_y,new_z]
		normal.append(points)

	steps.append(normal)
	#cycles_x.append(new_x)
	#cycles_y.append(new_y)
	#cycles_z.append(new_z)

    if DEBUG:
	print_cycles(steps)

    print "==== COMPLETE ====\n"
    return steps

def print_cycles(cycles):
    #global time
    #global acc_x
    #global acc_y
    #global acc_z
    #global peaks

    print "Writing current data to:   normalized.csv\n"

    out = open("normalized.csv", 'w')

    header = "index,i,acc_x,acc_y,acc_z\n"
    out.write(header)

    for index in range(len(cycles)):
	for i in range(0,STANDARD_LENGTH):
	    x = cycles[index][0]
	    y = cycles[index][1]
	    z = cycles[index][2]

	    output = str(index) + ',' + str(i) + ',' + str(x) + ',' + str(y) + ',' + str(z) +'\n'
#	    print output
	    out.write(output)

    out.close()

def build_avg_cycle(cycles):
    #CREATE AVG GAIT CYCLE FOR USER BY TAKING AVERAGE OF EACH POINT IN STEPS

    template = []
    for i in range(0,STANDARD_LENGTH):
	sum_x = 0
	sum_y = 0
	sum_z = 0
	for index in range(len(cycles)):
	    sum_x += cycles[index][i][0]
	    sum_y += cycles[index][i][1]
	    sum_z += cycles[index][i][2]

	avg_x = sum_x / len(cycles)
	avg_y = sum_y / len(cycles)
	avg_z = sum_z / len(cycles)
	
	points = [avg_x, avg_y, avg_z]
	template.append(points)
	
    return template

def create_template(template,filename):
    #CREATE TEMPLATE FOR USER BY TAKING AVERAGE OF EACH POINT IN STEPS

    out = open(filename, 'w')
    header = "index,acc_x,acc_y,acc_z\n"
    out.write(header)

    for index in range(len(template)):
	output = str(index) + ',' + str(template[index][0]) + ',' + str(template[index][1]) + ',' + str(template[index][2]) +'\n'
	#print output
	out.write(output)

    out.close()

def render_decision(probe1,probe2,template):

    print "Determining Match Score.."

    ## read template ## 
    with open(template, "rU") as infile:
	reader = csv.reader(infile)
	count = 0
	for line in reader:
	    if count == 0:
		count+=1
		continue 
	    gallary_x.append(float(line[1]))
	    gallary_y.append(float(line[2]))
	    gallary_z.append(float(line[3]))


    ## determine match ## 
    score1 = match(probe1, gallary_y)
    score2 = match(probe2, gallary_y)
    print "Score #1: "+str(score1)
    print "Score #2: "+str(score2)
    score = max(score1, score2)

    if score > THRESHOLD:
	print "MATCH FOUND: "+str(score)+"/"+str(STANDARD_LENGTH)
    else:
	print "NO MATCH FOUND: "+str(score)+"/"+str(STANDARD_LENGTH)

    return score

def match(probe, gal):

    count = 0
    for i in range(0,STANDARD_LENGTH):
	pr = probe[i][1]
	if pr > gal[i]-INTERVAL and pr < gal[i]+INTERVAL:
	    count+=1

    return count

def find_file(name, path):
    for root, dirs, files in os.walk(path):
	if name in files:
	    return True
	else:
	    return False

def main(argv):
 
    input = argv[0]
    outfile = argv[1]

    ## PARSE DATA ##
    parse_data(input)

    ## DATA TRANSFORMATION ##
    if TRANSFORM:
	modify_data()    

    ## TEMPLATE CREATION/MATCHING ##
    if find_file(outfile,DIR):
	print "== TEMPLATE EXISTS, MATCHING..."
	local_max()
	identify_dual_cycles()
	pattern1 = normalize(markers)
	pattern2 = normalize(markers_2)
	probe1 = build_avg_cycle(pattern1)
	probe2 = build_avg_cycle(pattern2)
	score = render_decision(probe1, probe2, DIR+outfile)

	if DISTRIBUTION:
	    text = str(score)+"\n"
	    out = open("distribution.csv", 'a')
	    out.write(text)
	    out.close()

    else:
	print "== TEMPLATE DOES NOT EXIST, CREATING FOR "+outfile+"..."
	local_max()
	identify_cycles()
	pattern = normalize(markers)
	gallary = build_avg_cycle(pattern)
	create_template(gallary, DIR+outfile)
	print "==== TEMPLATE CREATED ===="

 
if __name__ == "__main__":
    main(sys.argv[1:])
