#!usr/bin/python
 
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import sys, csv, os
from math import pow,sqrt,sin,cos,acos,pi

# Define Constants
PI = pi
DIR = "Templates/"
STANDARD_LENGTH = 100
INTERVAL = 0.05
THRESHOLD = STANDARD_LENGTH-20
RANGE = 17
TRIM = 0.10
DEBUG = True
TRANSFORM = True
FIXED = True
DISTRIBUTION = True

# Global Variables
### START HERE: REDUCE GLOBAL VARIABLES
average = 0.0
count = 0
time = []
vertaccel = []	#fl
distance = []	#fl
orig_accel = []
accel = []
gravity = []
peaks = []	#possibly make local
markers = []	#make local
markers_2 = []	#make local
gallary = []
deviation = []
Rx = [[0 for x in range(3)] for x in range(3)]
Ry = [[0 for x in range(3)] for x in range(3)]
Rz = [[0 for x in range(3)] for x in range(3)]


def parse_data(input):
    global average
    global count
    global vertaccel

    print "\n==== READING DATA FILE ===="

    ## read data ## 
    with open(input, "rU") as infile:
	reader = csv.reader(infile)
	sum = 0.0
	for line in reader:
	    if count == 0:
		count+=1
		continue 
	    time.append(line[0]);
	    #x_acc.append(float(line[1]));
	    #y_acc.append(float(line[2]));
	    #z_acc.append(float(line[3]));
	    accel_points = [float(line[1]), float(line[2]), float(line[3])]
	    orig_accel.append(accel_points)
	    
	    sum += float(line[2])
	    count+=1
	    
	    accel_points = [float(line[1]), float(line[2]), float(line[3])]
	    gravity.append([float(line[4]), float(line[5]), float(line[6])])

	average = abs(sum / count)
	count = count - 1
	print "Number of Sample Points: "+str(count)
	print "Average y_acceleration: "+str(average)

    x = np.loadtxt(input,delimiter=',', skiprows=1) #fl
    vertaccel = np.diag(np.dot(x[:,1:3],x[:,4:6].T)) #fl
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

    global count

    if DEBUG:
	print "Writing current data to:   modified.csv\n"

	out = open("Debug/modified.csv", 'w')
	
	#header = "time,acc_x,acc_y,acc_z,gravity_x,gravity_y,gravity_z\n"
	header = "time,acc_x,acc_y,acc_z,gravity_x,gravity_y,gravity_z,vertaccel,corr\n"
	out.write(header)

    for index in range(count):
	# Define local variables
	acc_x = orig_accel[index][0]
	acc_y = orig_accel[index][1]
	acc_z = orig_accel[index][2]
	grav_x = gravity[index][0]
	grav_y = gravity[index][1]
	grav_z = gravity[index][2]

	# Create 3x1 matrix of acceleration and gravity values
	vertices = [[acc_x], [acc_y], [acc_z]]
	grav = [[grav_x], [grav_y], [grav_z]]

	#print "before rotation"
	#print vertices
	#print gravity
	
	if DEBUG:
	    output = time[index] + ','

	# ROTATE ON X-AXIS
	#grav_xy = sqrt(pow(grav_x,2) + pow(grav_y,2))
	#grav_xz = sqrt(pow(grav_x,2) + pow(grav_z,2))
	grav_yz = sqrt(pow(grav_y,2) + pow(grav_z,2))

	angx = float(acos(grav_y / grav_yz))
	#angy = float(acos(grav_y / grav_xz))
	#angz = float(acos(grav_y / grav_xy))

	# Determine quadrant vector is in, modify angle if necessary
	if grav_y > 0: 
	    if grav_z > 0: 
		angx = PI - angx
	    else:
		angx = PI + angx
	else:
	    if grav_z > 0: 
		angx = (2*PI) - angx


	# Perform Rotation
	#print "Rotate on X-axis by "+str(angx)+" radians"
	build_rotation_matrix(angx, 0, 0)	#Rotation matrix for x only
	acc1 = matrix_multiplication(Rx, vertices)
	grav1 = matrix_multiplication(Rx, grav)

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

	accel.append([float(acc2[0][0]), float(acc2[1][0]), float(acc2[2][0])])

	if DEBUG:
	    output += str(acc2[0][0]) + ',' + str(acc2[1][0]) + ',' + str(acc2[2][0]) + ',' 
	    output += str(grav2[0][0]) + ',' + str(grav2[1][0]) + ',' + str(grav2[2][0]) + ',' 
	    output += str(vertaccel[index])	#fl
	    output += '\n'
	    out.write(output)

    print "==== COMPLETE ====\n"

def opt_distance(data):	#TEMP NAME

    DEFAULT = 105
    corr = []
    for lag in xrange(0,450):
	corr.append(correlation(data,offset=250,winsize=100,lag=lag))

    if DEBUG:
	out = open("Debug/correlation.csv", 'w')

	header = "index,corr_coef\n"
	out.write(header)

	for index in range(len(corr)):
	    output = str(index) + ',' + str(corr[index]) +'\n'
	    out.write(output)

	out.close()
    
    corr_values = local_max(corr,base=False,filename="distance.csv")

    for index in range(len(corr_values)):
	if corr_values[index] == 0:
	    continue
	else:
	    return index
	    
    # if it gets to this point, no correlation can be determined, return default value
    return DEFAULT


def correlation(data, offset=0, winsize=2, lag=0):
    # CALCULATE THE PEARSON COORELATION COEFFICIENT
    #   will be used to determine optimal gait length

    return stats.pearsonr(data[offset:offset+winsize], data[offset+lag:offset+lag+winsize])[0]


def local_max(data, base=True, filename="local_max.csv"):
    # CALCULATE THE LOCAL MAXIMUMS IN THE ACC_Y VALUES
    # FILTERING CODE BASED ON 40-point window
    #     (point must be maximum of 20 points on either side)

    print "DETECTING LOCAL MAX POINTS"
    MAX_WINDOW = 28
    peaks = []

    entries = len(data)
    count = 0

    if base:
	thresh = average + 1
    else:
	thresh = 0.5

    print "Threshold: "+str(thresh)

    for x in range (MAX_WINDOW, entries-MAX_WINDOW):

	values = data[x-MAX_WINDOW:x+MAX_WINDOW+1]
	curr = data[x]
	max_y = max(values)

	if max_y == curr and curr > thresh:
	    peaks.append(curr)
	    count+=1
	else:
	    peaks.append(float(0))


    # Account for values skipped at begining/end of list
    zeros = [0] * MAX_WINDOW
    peaks = zeros + peaks + zeros

    print "Local Max Points: "+str(count)

    if DEBUG and base:
	output_local_max(peaks,filename=filename)
    elif DEBUG:
	output_corr_max(data,peaks,filename=filename)

    return peaks


def output_corr_max(data, peaks, filename):

    print "Writing current data to:   "+filename +"\n"

    file = "Debug/"+filename
    out = open(file, 'w')

    header = "index,corr_coef,local max\n"
    out.write(header)

    for index in range(len(data)):
	output = str(index) + ',' + str(data[index]) + ',' + str(peaks[index]) +'\n'
	#print output
	out.write(output)


def output_local_max(peaks,filename):

    print "Writing current data to:   "+filename +"\n"

    file = "Debug/"+filename
    out = open(file, 'w')

    header = "time,acc_x,acc_y,acc_z,local max\n"
    out.write(header)

    for index in range(count):
	output = time[index] + ',' + str(accel[index][0]) + ',' + str(accel[index][1]) + ',' + str(accel[index][2]) + ',' + str(peaks[index]) +'\n'
	#print output
	out.write(output)

    out.close()

# DIVIDE DATA INTO GAIT CYCLES
def identify_cycles(distance):
    counter = -2	#ignore first 2 points
    begin = 0
    mid = 0
    end = 0
    
    for index in range(len(peaks)):
	if peaks[index] == 0:
	    continue
	else:
	    counter += 1

	    if counter == 1:
		begin = index

	    elif counter == 2:
		mid = index

	    elif counter == 3:
		end = index
		     
		length = end-begin
		points = [begin,mid,end]	
		if length >= distance-RANGE and length <= distance+RANGE:
		    print "points: "+str(points)
		    markers.append(points)

		    counter = 1
		    begin = end	
		else:
		    # Check for missing local max point
		    print "Checking for missing local max point.."
		    points = [begin, (begin+mid)/2, mid]
		    length = mid-begin
		    if length >= distance-RANGE and length <= distance+RANGE:
			print "Missing point detected!"
			print "points: "+str(points)
			markers.append(points)

			counter = 2
			begin = mid
			mid = end
		    else:
			print "Length: "+str(length)
			print "Invalid Cycle Length.  Discarding..."+ str(points)
			counter = 0

    print "Cycles Detected: "+str(len(markers))
    return

# DIVIDE DATA INTO GAIT CYCLES
def identify_dual_cycles(distance):
    counter_1 = -2	#ignore first two points
    begin_1 = 0
    mid_1 = 0
    end_1 = 0

    counter_2 = -3	#ignore first two points
    begin_2 = 0
    mid_2 = 0
    end_2 = 0
    
    for index in range(len(peaks)):
	if peaks[index] == 0:
	    continue
	else:

	    counter_1 += 1
	    counter_2 += 1
	    #print "counter_1: "+str(counter_1)
	    #print "counter_2: "+str(counter_2)

	    # 'a' cycles
	    if counter_1 == 1:
		begin_1 = index

	    elif counter_1 == 2:
		mid_1 = index

	    elif counter_1 == 3:
		end_1 = index
		     
		length = end_1-begin_1
		points = [begin_1,mid_1,end_1]	
		if length >= distance-RANGE and length <= distance+RANGE:
		    print "a points: "+str(points)
		    markers.append(points)

		    counter_1 = 1
		    begin_1 = end_1	
		else:
		    # Check for missing local max point
		    print "a Length: "+str(length)
		    print "Invalid Cycle Length. Checking for missing local max point.."
		    points_alt = [begin_1, (begin_1+mid_1)/2, mid_1]
		    length = mid_1-begin_1

		    if length >= distance-RANGE and length <= distance+RANGE:
			print "Missing point detected!"
			print "old points: "+str(points)
			print "new points: "+str(points_alt)
			markers.append(points)

			# reset counters
			counter_2 = 1
			counter_1 = 2

			# adjust points
			begin_2 = end_1
			begin_1 = mid_1
			mid_1 = end_1
			continue
		    else:
			print "a Length: "+str(length)
			print "Invalid Cycle Length.  Discarding..."+str(points)
			counter_1 = 1
			begin_1 = end_1

	    # 'b' cycles
	    if counter_2 == 1:
		begin_2 = index

	    elif counter_2 == 2:
		mid_2 = index

	    elif counter_2 == 3:
		end_2 = index
		
		length = end_2-begin_2
		points = [begin_2,mid_2,end_2]	
		if length >= distance-RANGE and length <= distance+RANGE:
		    print "b points: "+str(points)
		    #print points
		    markers_2.append(points)

		    counter_2 = 1
		    begin_2 = end_2
		else:
		    # Check for missing local max point
		    print "b Length: "+str(length)
		    print "Invalid Cycle Length. Checking for missing local max point.."
		    points_alt = [begin_2, (begin_2+mid_2)/2, mid_2]
		    length = mid_2-begin_2

		    if length >= distance-RANGE and length <= distance+RANGE:
			print "Missing point detected!"
			print "old points: "+str(points)
			print "new points: "+str(points_alt)
			markers.append(points_alt)

			# reset counters
			counter_1 = 1
			counter_2 = 2

			# adjust points
			begin_1 = end_2
			begin_2 = mid_2
			mid_2 = end_2
		    else:
			print "b Length: "+str(length)
			print "Invalid Cycle Length.  Discarding..."+ str(points)
			counter_2 = 1
			begin_2 = end_2

    print "'a' Cycles Detected: "+str(len(markers))
    print "'b' Cycles Detected: "+str(len(markers_2))+"\n"
    return

def normalize(marks, filename):
    # NORMALIZE STEPS
    # AMPLITUDE - DIVIDE BY MAX Y-VALUE
    # LENGTH - HAVE COMMON LENGTH (100 data points) AND FIT TO THAT LENGTH USING LINEAR INTERPOLATION
    
    print "Normalizing cycles..."

    template = []
    steps = []

    # Normalize Length
    for index in range(len(marks)):

	begin = marks[index][0]
	mid = marks[index][1]
	end = marks[index][2]

	#get points in cycle
	x = [col[0] for col in accel][begin:end+1]
	y = [col[1] for col in accel][begin:end+1]
	z = [col[2] for col in accel][begin:end+1]

	#adjust markers
	end = end - begin
	mid = mid - begin
	begin = 0

	steps.append(normalize_length(x,y,z,end))

    if DEBUG:
	print_cycles(steps, filename)

    # Build Average Cycle
    averages = build_avg_cycle(steps)

    # Normalize Height
    old_x = [col[0] for col in averages]
    old_y = [col[1] for col in averages]
    old_z = [col[2] for col in averages]

    new_x = normalize_height(old_x)
    new_y = normalize_height(old_y)
    new_z = normalize_height(old_z)

    #Standard Deviation
    avg_x = sum(new_x) / len(new_x)
    avg_y = sum(new_y) / len(new_y)
    avg_z = sum(new_z) / len(new_z)

    for i in range(0,STANDARD_LENGTH):
#	var_x = pow((new_x[i] - avg_x),2)
#	var_y = pow((new_y[i] - avg_y),2)
#	var_z = pow((new_z[i] - avg_z),2)

#	dev_x = sqrt(var_x / length)
#	dev_y = sqrt(var_y / length)
#	dev_z = sqrt(var_z / length)

	dev_x = sqrt( pow( (new_x[i] - avg_x), 2) / len(new_x) )
	dev_y = sqrt( pow( (new_y[i] - avg_y), 2) / len(new_y) )
	dev_z = sqrt( pow( (new_z[i] - avg_z), 2) / len(new_z) )

	points = [new_x[i], new_y[i], new_z[i], dev_x, dev_y, dev_z]
	template.append(points)


    print "==== COMPLETE ====\n"

    if DEBUG:
	new_file = "Debug/template_"+filename
	create_template(template,new_file)	

    return template



def normalize_height(data):

    print "Normalizing Height.."
    new_data = [0] * STANDARD_LENGTH

    maxval = max(data)    
    minval = min(data)    
    #print sorted(data)
    print "Max: "+str(maxval)
    print "Min: "+str(minval)
    span = maxval - minval
    print "Span: "+str(span)+"\n"

    for i in range(0,STANDARD_LENGTH):
	new_data[i] = (data[i] - minval) / span 

    return new_data


def normalize_length(x, y, z, end):

    #normalize length
    normal = []

    length = float(end) / (STANDARD_LENGTH-1)
    interval = 0
    for i in range(0,STANDARD_LENGTH):
	if i == 0:
	    points = [x[0],y[0],z[0]]	
	    normal.append(points)

	elif i == STANDARD_LENGTH-1:
	    points = [x[end],y[end],z[end]]	
	    normal.append(points)

	else:
	    interval = length*i
	    prev = int(interval)
	    next = prev+1
	    gap = interval - prev

	    diff_x = x[next] - x[prev]
	    diff_y = y[next] - y[prev]
	    diff_z = z[next] - z[prev]

	    new_x = x[prev] + (gap * diff_x)
	    new_y = y[prev] + (gap * diff_y)
	    new_z = z[prev] + (gap * diff_z)

	    points = [new_x,new_y,new_z]
	    normal.append(points)

    return normal


def print_cycles(cycles, filename):

    print "Writing current data to:   "+ filename +"\n"

    file = "Debug/"+filename
    out = open(file, 'w')

    header = "index,i,acc_x,acc_y,acc_z\n"
    out.write(header)

    for index in range(len(cycles)):
	for i in range(0,STANDARD_LENGTH):
	    x = cycles[index][i][0]
	    y = cycles[index][i][1]
	    z = cycles[index][i][2]

	    output = str(index) + ',' + str(i) + ',' + str(x) + ',' + str(y) + ',' + str(z) +'\n'
#	    print output
	    out.write(output)

    out.close()


def build_avg_cycle(cycles):
    #CREATE AVG GAIT CYCLE FOR USER BY TAKING AVERAGE OF EACH POINT IN STEPS

    averages = []

    #Get amount of samples to trim from average
    buffer = int(len(cycles) * TRIM)	
    length = len(cycles) - (2*buffer)
    print "Buffer: "+str(buffer)+"\n"
    print "Length: "+str(len(cycles))+"\n"
    print "new Length: "+str(length)+"\n"

    for i in range(0,STANDARD_LENGTH):
	# Reset Variables
	x = []
	y = []
	z = []
	sum_x = 0.0
	sum_y = 0.0
	sum_z = 0.0

	for index in range(len(cycles)):
	    x.append(cycles[index][i][0])
	    y.append(cycles[index][i][1])
	    z.append(cycles[index][i][2])

	x.sort()
	y.sort()
	z.sort()

	sum_x = sum(x[buffer:-buffer])
	sum_y = sum(y[buffer:-buffer])
	sum_z = sum(z[buffer:-buffer])

	avg_x = sum_x / length
	avg_y = sum_y / length
	avg_z = sum_z / length

	points = [avg_x, avg_y, avg_z]
	averages.append(points)

    return averages 



def create_template(template,filename):
    #CREATE TEMPLATE FOR USER BY TAKING AVERAGE OF EACH POINT IN STEPS

    out = open(filename, 'w')
    header = "index,acc_x,acc_y,acc_z,dev_x,dev_y,dev_z\n"
    out.write(header)

    for index in range(len(template)):
	output = str(index) + ',' + str(template[index][0]) + ',' + str(template[index][1]) + ',' + str(template[index][2]) + ',' + str(template[index][3]) + ',' + str(template[index][4]) + ',' + str(template[index][5]) +'\n'
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
	    gallary.append([float(line[1]), float(line[2]), float(line[3])])
	    deviation.append([float(line[4]), float(line[5]), float(line[6])])


    ## Load data
    gallary_y = [col[1] for col in gallary]

    if FIXED:
	inter = [INTERVAL] * STANDARD_LENGTH
    else:
	inter = [col[1] for col in deviation]

    dataset_1 = []
    dataset_2 = []
    for i in range(0,STANDARD_LENGTH):
	dataset_1.append(probe1[i][1])
	dataset_2.append(probe2[i][1])

    temp1 = stats.pearsonr(dataset_1, gallary_y)[0] * 100
    temp2 = stats.pearsonr(dataset_2, gallary_y)[0] * 100
    print "Temp #1: "+str(temp1)
    print "Temp #2: "+str(temp2)

    ## determine match ## 
    score1 = match(probe1, gallary_y, inter)
    score2 = match(probe2, gallary_y, inter)
    print "Score #1: "+str(score1)
    print "Score #2: "+str(score2)
    score =  int(max(temp1, temp2))
    #score = max(score1, score2)

    if score > THRESHOLD:
	print "MATCH FOUND: "+str(score)+"/"+str(STANDARD_LENGTH)
    else:
	print "NO MATCH FOUND: "+str(score)+"/"+str(STANDARD_LENGTH)

    return score


def match(probe, gal, sd):

    count = 0
    for i in range(0,STANDARD_LENGTH):
	pr = probe[i][1]
	dev = sd[i]
	if pr >= gal[i]-dev and pr <= gal[i]+dev:
	    count+=1

    return count


def find_file(name, path):
    for root, dirs, files in os.walk(path):
	if name in files:
	    return True
	else:
	    return False


def main(argv):
 
    global peaks
    input = argv[0]
    outfile = argv[1]

    ## PARSE DATA ##
    parse_data(input)

    ## DATA TRANSFORMATION ##
    if TRANSFORM:
	modify_data()    
    acc_y = [col[1] for col in accel] 
    print len(acc_y)

    global distance
    distance = opt_distance(acc_y)
    print "Optimal Distance: "+str(distance) + "\n"

    ## TEMPLATE CREATION/MATCHING ##
    if find_file(outfile,DIR):
	print "== TEMPLATE EXISTS, MATCHING..."
	peaks = local_max(acc_y)
	identify_dual_cycles(distance)
	probe1 = normalize(markers, "normalize_a.csv")
	probe2 = normalize(markers_2, "normalize_b.csv")
	score = render_decision(probe1, probe2, DIR+outfile)

	if DISTRIBUTION:
	    text = str(score)+"\n"
	    out = open("distribution.csv", 'a')
	    out.write(text)
	    out.close()

    else:
	print "== TEMPLATE DOES NOT EXIST, CREATING FOR "+outfile+"..."
	peaks = local_max(acc_y)
	identify_cycles(distance)
	gallary = normalize(markers, "normalized.csv")
	create_template(gallary, DIR+outfile)
	print "==== TEMPLATE CREATED ===="

 
if __name__ == "__main__":
    main(sys.argv[1:])
