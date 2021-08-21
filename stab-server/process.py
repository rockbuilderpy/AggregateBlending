import pandas as pd
import matplotlib
from shapely.geometry import Polygon

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np

def percent_passing(sample):  # Calculates Cumulative Percent Passing

    percent_retained = []
    per_pass = []
    for i in range(sample.shape[0]):
        percent_retained.append(sample.iloc[i, 1] * 100 / sample.sum()[1])
        per_pass.append(100 - sum(percent_retained))
    sample['Cumulative percent passing'] = per_pass

def best_fit(sample):  # Finds The BestFit 'LINEAR STRAIGHT LINE' Parameters For The Given Curve

    modif_theta = []
    data_fr = reshape_matrix(sample)
    x_value = np.array(data_fr.iloc[0:, 0:2])
    y_value = np.transpose(np.array([data_fr.iloc[0:, 3]]))
    theta = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(x_value), x_value)), np.transpose(x_value)),
                      y_value)
    for sublist in theta:
        for item in sublist:
            modif_theta.append(item)
    return modif_theta

def reshape_matrix(sample):     # ReFormats DataFrame To Add A Bias Term (Just for Computational Purposes)
    bias = []
    for i in range(sample.shape[0]):
        bias.append(1)
    df = pd.DataFrame({'bias': bias})
    df = pd.concat([df, sample], axis=1)
    return df

def extract_sample(sample):     # ReFormats DataFrame Such That The Least Possible 100% Passing Sieve Is Obtained
    i = 0
    sample = np.array(sample.iloc[0:, 0:3])
    while i < sample.shape[0]:
        if sample[i + 1][2] != 100:
            break
        for j in range(2):
            if sample[i][2] == 100:
                sample = np.delete(sample, i, axis=0)
                break
    sample = pd.DataFrame(sample, columns=['Sieve Size (mm)', 'Weight Retained (gm)', 'Cumulative percent passing'])
    return sample

def transform_by(sample):      # Returns The New Translated X-Coordinate Of Origin
    grade_used = sample.iloc[0:, 3]
    origin_X = (grade_used - sample.iloc[0:, 0]).tolist()
    return origin_X

def test_transform(sample):    # Transformed X-Coordinates Of Sieve's
    origin = transform_by(sample)
    sample.iloc[0:, 0] = sample.iloc[0:, 0] + origin
    return sample

def polygon_builder(sample, str_line_2):    # Cost Function
    xy_values = []
    line_slo_inter = []
    last_two_points = False
    polygon_coor = []
    inter_coor =[]

    for i in range(sample.shape[0] - 1):
        xy_values.append([sample.iloc[i, 0], sample.iloc[i, 2], sample.iloc[i + 1, 0], sample.iloc[i + 1, 2]])
        line_slo_inter.append(slope_intercept(xy_values[i][0], xy_values[i][1], xy_values[i][2], xy_values[i][3]))
    for i in range(len(line_slo_inter)):
        inter_sec = inter_point(line_slo_inter[i], str_line_2)
        if sample.iloc[i+1, 0] <= inter_sec[0] <= sample.iloc[i, 0]:
            inter_coor.append(inter_sec)
    inter_coor.append([0,0])
    inter_coor.append([predicted_X(100, str_line_2), 100])
    if inter_coor[len(inter_coor) - 1][0] < sample.iloc[0, 0]:
        inter_coor.append([sample.iloc[0, 0], sample.iloc[0, 2]])
    inter_coor = sorted(inter_coor, key=lambda x: x[0])
    best_fit_inter = [[predicted_X(0, str_line_2), 0],[predicted_X(100, str_line_2), 100]]
    for i in range(len(inter_coor) - 1):  # Outer Loop Runs On [ [0,0], [Intersection of Co-ordinates With Distribution Curve], [x,100]
        if last_two_points:
            break
        temp_coor = [inter_coor[i], inter_coor[i+1]]
        for j in range(sample.shape[0]):  # Inner Loop Runs Within Sample Sieve Sizes
            if inter_coor[i][0] < sample.iloc[j, 0] < inter_coor[i + 1][0]:
                temp_coor.append([sample.iloc[j, 0], sample.iloc[j, 2]])
        if inter_coor[i][0] < best_fit_inter[0][0] < inter_coor[i + 1][0]:
            temp_coor.append(best_fit_inter[0])
        if inter_coor[i][0] < best_fit_inter[1][0] < inter_coor[i + 1][0]:
            temp_coor.append(best_fit_inter[1])
        if best_fit_inter[0][0] < 0 and inter_coor[i][0] == 0:  # Assuming That There Would Be No Negative Value For y = 100 for best_fit_line
            temp_coor.append(best_fit_inter[0])
        temp_coor = sorted(temp_coor, key=lambda x: x[0])
        if len(temp_coor) == 2:
            last_two_points = True
            temp_coor.append([sample.iloc[0, 0], sample.iloc[0, 2]])
        if best_fit_inter[0][0] < 0 and inter_coor[i][0] == 0:
            temp_coor.append(best_fit_inter[0])
        else:
            temp_coor.append(inter_coor[i])
        polygon_coor.append(temp_coor)
    #print 'Polygon Coordinates', polygon_coor
    polygon_area = calculate_area(polygon_coor, sample, str_line_2)
    #print inter_coor
    #print polygon_area
    #print sum(polygon_area)
    return sum(polygon_area)

def calculate_area(polygon_coor, sample, str_line):
    polygon_area=[]
    for i in range(len(polygon_coor)):
        value = 0
        sign = +1
        for j in range(sample.shape[0]):
            for k in range(len(polygon_coor[i])):
                if polygon_coor[i][k][0] == sample.iloc[j, 0]:
                    value = [polygon_coor[i][k][0], sample.iloc[j, 2]]
                    break
            if value != 0:
                break
        pred_y_line = str_line[0] * value[0] + str_line[1]
        if pred_y_line < value[1]:
            sign = -1
        polygon = Polygon(polygon_coor[i])
        polygon_area.append(polygon.area * sign)
    return polygon_area

def optimize_area(theta, sample, learning_rate = 0.0001, iterations = 350):
    cost_history = []
    print('Theta Before: ', theta)
    while True:
        cost_history.append(polygon_builder(sample, theta))
        theta[0] = theta[0] - (learning_rate * float(cost_history[len(cost_history) - 1]))
        theta[1] = theta[1] - (learning_rate * float(cost_history[len(cost_history) - 1]))
        if -1.0 < cost_history[len(cost_history) - 1] < 1.0:
            break
    print(cost_history)
    print('Theta After', theta)
    return theta

def slope_intercept(x1,y1,x2,y2):
    a = (y2 - y1) / (x2 - x1)
    b = y1 - a * x1
    return a,b

def inter_point(str_1, str_2):
    x = (str_2[1] - str_1[1]) / (str_1[0] - str_2[0])
    y = str_1[0] * x + str_1[1]
    return [x,y]

def predicted_X(y_value, theta):
    pred_x = (y_value - theta[1]) / theta[0]
    return pred_x

def join_best_fit(theta_A, theta_B, theta_C):
    theta_A_modif = np.array(theta_A)
    theta_B_modif = np.array(theta_B)
    theta_C_modif = np.array(theta_C)
    theta_A_modif = np.flipud(theta_A_modif)
    theta_B_modif = np.flipud(theta_B_modif)
    theta_C_modif = np.flipud(theta_C_modif)
    coor_A = [[predicted_X(0, theta_A_modif), 0], [predicted_X(100, theta_A_modif), 100]]
    coor_B = [[predicted_X(0, theta_B_modif), 0], [predicted_X(100, theta_B_modif), 100]]
    coor_C = [[predicted_X(0, theta_C_modif), 0], [predicted_X(100, theta_C_modif), 100]]
    slo_inter = [slope_intercept(coor_C[1][0], coor_C[1][1], coor_B[0][0], coor_B[0][1]),
                 slope_intercept(coor_B[1][0], coor_B[1][1], coor_A[0][0], coor_A[0][1])]
    return [[coor_C[1][0], coor_C[1][1]], [coor_B[0][0], coor_B[0][1]],[coor_B[1][0], coor_B[1][1]], [coor_A[0][0], coor_A[0][1]]], slo_inter

def get_max_slope_inter(sample):
    slope_inter = []
    for i in range(sample.shape[0] - 1):
        slope_inter.append(slope_intercept(sample.iloc[i, 0], sample.iloc[i, 2], sample.iloc[i + 1, 0], sample.iloc[i + 1, 2]))
    slope_inter = sorted(slope_inter, key=lambda x: x[0])
    return list(slope_inter[len(slope_inter) - 1])

class AggregateMixProportion:

    def _init_(self):
        data = pd.read_excel(r'/Users/nikhil/Desktop/Project/Blending Of Aggregate/Utilities/Input.xlsx')
        self.sample_A = data.iloc[0:, [0, 1]]   # Sample A
        self.sample_B = data.iloc[0:, [0, 2]]   # Sample B
        self.sample_C = data.iloc[0:, [0, 3]]   # Sample C

        percent_passing(self.sample_A)  # Cumulative Percent A
        percent_passing(self.sample_B)  # Cumulative Percent B
        percent_passing(self.sample_C)  # Cumulative Percent C

        #self.sample_A.iloc[0:, 2] = [100, 63, 19, 8, 5, 3, 0, 0]
        #self.sample_B.iloc[0:, 2] = [100, 100, 100, 93, 55, 36, 3, 0]
        #self.sample_C.iloc[0:, 2] = [100, 100, 100, 100, 100, 97, 88, 0]

        self.sample_A = pd.concat([self.sample_A, data.iloc[0:, 4]], axis=1)    # Inculcates Grade Of Sample Used Into The DataFrame (Sample A)
        self.sample_B = pd.concat([self.sample_B, data.iloc[0:, 4]], axis=1)    # Inculcates Grade Of Sample Used Into The DataFrame (Sample B)
        self.sample_C = pd.concat([self.sample_C, data.iloc[0:, 4]], axis=1)    # Inculcates Grade Of Sample Used Into The DataFrame (Sample C)

        self.trans_sample_A = test_transform(self.sample_A)     # UnFormat DataFrame With Translated X-Coordinate's Of Sieve's (Sample A)
        self.trans_sample_B = test_transform(self.sample_B)     # UnFormat DataFrame With Translated X-Coordinate's Of Sieve's (Sample B)
        self.trans_sample_C = test_transform(self.sample_C)     # UnFormat DataFrame With Translated X-Coordinate's Of Sieve's (Sample C)

        self.extr_sample_A = extract_sample(self.trans_sample_A)  # ReFormat DataFrame Such That Only 100% Passing Sieve Is Obtained (Sample A)
        self.extr_sample_B = extract_sample(self.trans_sample_B)  # ReFormat DataFrame Such That Only 100% Passing Sieve Is Obtained (Sample A)
        self.extr_sample_C = extract_sample(self.trans_sample_C)  # ReFormat DataFrame Such That Only 100% Passing Sieve Is Obtained (Sample A)

        self.theta_A = best_fit(self.extr_sample_A)    # Obtained BestFit (Sample A)
        self.theta_B = best_fit(self.extr_sample_B)    # Obtained BestFit (Sample B)
        self.theta_C = best_fit(self.extr_sample_C)    # Obtained BestFit (Sample C)

        print(self.extr_sample_A.to_string())
        line_param = get_max_slope_inter(self.extr_sample_A)                    # Gets The Maximum Slope Line To Be Oriented
        self.theta_A = np.array(optimize_area(line_param, self.extr_sample_A))  # Gets The Minimum Balanced Area Line Parameters (Sample A)
        self.theta_A = list(reversed(self.theta_A))                             # Input Type For Other Line(s) Of Code
        print('------------------------------------------------------------------------------------')

        print(self.extr_sample_B.to_string())
        line_param = get_max_slope_inter(self.extr_sample_B)                     # Gets The Maximum Slope Line To Be Oriented
        self.theta_B = np.array(optimize_area(line_param, self.extr_sample_B))   # Gets The Minimum Balanced Area Line Parameters (Sample A)
        self.theta_B = list(reversed(self.theta_B))                              # Input Type For Other Line(s) Of Code
        print('------------------------------------------------------------------------------------')

        print(self.extr_sample_C.to_string())
        line_param = get_max_slope_inter(self.extr_sample_C)                    # Gets The Maximum Slope Line To Be Oriented
        self.theta_C = np.array(optimize_area(line_param, self.extr_sample_C))  # Gets The Minimum Balanced Area Line Parameters (Sample A)
        self.theta_C = list(reversed(self.theta_C))                             # Input Type For Other Line(s) Of Code
        print('------------------------------------------------------------------------------------')

        self.join_bes_fit, self.slo_inter = join_best_fit(self.theta_A, self.theta_B,self.theta_C)  # Obtains Co-Ordinates Of Intersected Best-Fit Line
        self.inter_1 = inter_point(self.slo_inter[0],[1, 0])                                        # Intersected Co-ordinates of Line Joining Best-fit With X=Y line
        self.inter_2 = inter_point(self.slo_inter[1],[1, 0])                                        # Intersected Co-ordinates of Line Joining Best-fit With X=Y line

        self.resh_sample_A = reshape_matrix(self.trans_sample_A)    # Included Bias Term Computational Purpose (Sample A)
        self.resh_sample_B = reshape_matrix(self.trans_sample_B)    # Included Bias Term Computational Purpose (Sample B)
        self.resh_sample_C = reshape_matrix(self.trans_sample_C)    # Included Bias Term Computational Purpose (Sample C)

        self.straight_line_A = np.matmul(np.array(self.resh_sample_A.iloc[0:, 0:2]), np.array(self.theta_A))    # Predicted Value (Sample A)
        self.straight_line_B = np.matmul(np.array(self.resh_sample_B.iloc[0:, 0:2]), np.array(self.theta_B))    # Predicted Value (Sample B)
        self.straight_line_C = np.matmul(np.array(self.resh_sample_C.iloc[0:, 0:2]), np.array(self.theta_C))    # Predicted Value (Sample C)

        print()
        print('Percentage of Sample A :', self.inter_1[0])
        print('Percentage of Sample B :', self.inter_2[0] - self.inter_1[0])
        print('Percentage of Sample C :', 100 - self.inter_2[0])

    def plot_curve(self):
        plt.scatter(self.extr_sample_A.iloc[0:, 0], self.extr_sample_A.iloc[0:, 2])
        plt.scatter(self.extr_sample_B.iloc[0:, 0], self.extr_sample_B.iloc[0:, 2])
        plt.scatter(self.extr_sample_C.iloc[0:, 0], self.extr_sample_C.iloc[0:, 2])
        plt.plot(self.extr_sample_A.iloc[0:, 0], self.extr_sample_A.iloc[0:, 2], linewidth=1)
        plt.plot(self.extr_sample_B.iloc[0:, 0], self.extr_sample_B.iloc[0:, 2], linewidth=1)
        plt.plot(self.extr_sample_C.iloc[0:, 0], self.extr_sample_C.iloc[0:, 2], linewidth=1)
        plt.plot(plt.xlim(), plt.ylim(), color='black')
        plt.plot(self.trans_sample_A.iloc[0:, 0], self.straight_line_A, '--', color='grey')
        plt.plot(self.trans_sample_B.iloc[0:, 0], self.straight_line_B, '--', color='grey')
        plt.plot(self.trans_sample_C.iloc[0:, 0], self.straight_line_C, '--', color='grey')
        plt.plot([self.join_bes_fit[0][0], self.join_bes_fit[1][0]], [self.join_bes_fit[0][1],self.join_bes_fit[1][1]], '--', color='firebrick')
        plt.plot([self.join_bes_fit[2][0], self.join_bes_fit[3][0]], [self.join_bes_fit[2][1],self.join_bes_fit[3][1]], '--', color='firebrick')
        plt.plot([-5,self.inter_1[0]], [self.inter_1[1], self.inter_1[1]], '--', color='firebrick')
        plt.plot([-5,self.inter_2[0]], [self.inter_2[1], self.inter_2[1]], '--', color='firebrick')
        plt.axis([-5,120,-5,105])
        plt.xlabel('Sieve Size (mm)', color='black')
        plt.ylabel('Cumulative Percent Passing (%)', color='black')
        plt.legend(('Sample A', 'Sample B', 'Sample C'))
        plt.show()

obj = AggregateMixProportion()
obj.plot_curve()