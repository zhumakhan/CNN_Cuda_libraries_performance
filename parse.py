import os
import sys
import csv

filename = "log.txt"
metrics = "warp_execution_efficiency gld_efficiency	gst_efficiency	dram_utilization l2_utilization	shared_efficiency	shared_utilization	achieved_occupancy	ldst_fu_utilization cf_fu_utilization special_fu_utilization  tex_fu_utilization single_precision_fu_utilization double_precision_fu_utilization	stall_inst_fetch	stall_exec_dependency	stall_memory_dependency	stall_pipe_busy	stall_sync".split()


def reformat(val):
    ret = 0.0
    if type(val) == type('str'):
        try:
            if val[0] == '(':
                ret = float(val[1])/10
            elif val[-1] == '%':
                ret = float(val[:-1])/100
            else:
                ret = float(val)
        except:
            return val
    else:
        ret = val

    return round(ret, 2)

def parse_kernel_detailed(kernel):
    with open(filename, 'r') as f, open('dataset.csv','a+') as csvfile:
        line = f.readline()
        start = False
        count = 162
        csv_row = [kernel]
        while line and count > 0:
            if not start and kernel in line:
                start = True
                continue
            if start:
                count -= 1
                values = line.split()
                if values[1] in metrics:
                    print(values[1], values[-1])
                    csv_row.append(reformat(values[-1]))
            line = f.readline()
    return csv_row

def parse_time(kernel):
    with open('log.txt', 'r') as f:
        line = f.readline()
        while line:
            if kernel in line:
                average = line.split()[3 + 2*int(line.split()[0]=='GPU')]
                print(average)
                average = average[:-2]
                return reformat(average)

            line = f.readline()

def write_csv(row):
    with open('dataset.csv','a+') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',')
        row = list(map(reformat, row))
        csv_writer.writerow(row)


def combine(*rows):
    sum = 0.0
    n = 0
    for row in rows:
        print(row)
        sum += row[-1]
        n = len(row)

    res = [0.0]*n

    for row in rows:
        for i in range(1, n-1):
            res[i] += row[i]*row[-1]/sum
        res[-1] += row[-1]
    
    return res


kernel = "trt_maxwell_scudnn_winograd_128x128_ldg1_ldg4_relu_tile148n_nt_v0"
os.system("nvprof --profile-from-start off --metrics all --log-file log.txt ./trt_l35")
csv_row = parse_kernel_detailed(kernel)
os.system("nvprof --profile-from-start off --log-file log.txt ./trt_l35")
csv_row.append(parse_time(kernel))
csv_row[0] = 'trt'
write_csv(csv_row)



kernel_1 = "maxwell_sgemm_128x128_nn"
kernel_2 = "im2col_gpu_kernel_ext(int,"
kernel_3 = "add_bias_kernel(float*,"
os.system("cd darknet && nvprof --profile-from-start off --metrics all --log-file ../log.txt ./cublas_35 detect cfg/yolov4-tiny.cfg yolov4-tiny.weights dog.jpg")
csv_row_1 = parse_kernel_detailed(kernel_1)
csv_row_2 = parse_kernel_detailed(kernel_2)
csv_row_3 = parse_kernel_detailed(kernel_3)
os.system("cd darknet && nvprof --profile-from-start off --log-file ../log.txt ./cublas_35 detect cfg/yolov4-tiny.cfg yolov4-tiny.weights dog.jpg")

csv_row_1.append(parse_time(kernel_1))
csv_row_2.append(parse_time(kernel_2))
csv_row_3.append(parse_time(kernel_3))

csv_row = combine(csv_row_1, csv_row_2, csv_row_3)
csv_row[0] = 'cublas'
write_csv(csv_row)

kernel_1 = "maxwell_scudnn_winograd_128x128_ldg1_ldg4_mobile_relu_tile148t_nt_v0"
kernel_2 = "add_bias_kernel(float*,"
kernel_3 = "cudnn::winograd::generateWinogradTilesKernel<int=1,"
os.system("cd darknet && nvprof --profile-from-start off --metrics all --log-file ../log.txt ./cudnn_35 detect cfg/yolov4-tiny.cfg yolov4-tiny.weights dog.jpg")
csv_row_1 = parse_kernel_detailed(kernel_1)
csv_row_2 = parse_kernel_detailed(kernel_2)
csv_row_3 = parse_kernel_detailed(kernel_3)
os.system("cd darknet && nvprof --profile-from-start off --log-file ../log.txt ./cudnn_35 detect cfg/yolov4-tiny.cfg yolov4-tiny.weights dog.jpg")
csv_row_1.append(parse_time(kernel_1))
csv_row_2.append(parse_time(kernel_2))
csv_row_3.append(parse_time(kernel_3))

csv_row = combine(csv_row_1, csv_row_2, csv_row_3)
csv_row[0] = 'cudnn'
write_csv(csv_row)


