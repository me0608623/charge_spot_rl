
from typing import List
from icecream import ic
import wandb
from copy import deepcopy
x = [
		[591, 330, 22, 59, 82, 520, 286],
        [1176, 604, 25, 54, 4, 1069, 2, 21, 350, 107, 1023],
        [880, 17, 210, 1, 2, 26, 60, 23, 18, 100, 12, 90, 120, 31, 97, 141, 14, 154, 1800]
	]
y = [
    	[186, 572, 206, 203, 596, 19, 108],
        [415, 293, 121, 217, 1278, 21, 38, 826, 286, 221, 719],
        [262, 426, 47, 225, 38, 348, 102, 18, 238, 182, 561, 16, 81, 33, 337, 4, 48, 103, 727]
	]
z = [
    	[303, 412, 197, 478, 268, 53, 179],
        [200, 192, 455, 11, 498, 505, 31, 1287, 372, 506, 378],
        [537, 102, 193, 81, 409, 218, 143, 3, 100, 314, 264, 260, 193, 17, 78, 31, 35, 429, 389]
	]
x_range = [-1/3, 1/3]
y_range = [0.8 * -1/3, 0.8 * 1/3]
z_range = [-15, 15]
name = ['action 7', 'action 11', 'action 19']


def normal_list(input: List[List[int]]) -> List[List[float]]:
	output = deepcopy(input)  
	for  x_dim  in range(len(input)):
		count = 0
		for all_data in range(len(input[x_dim])):
			count += input[x_dim][all_data]
		print(count)
		for all_data in range(len(input[x_dim])):
			output[x_dim][all_data] = input[x_dim][all_data] / count
	return output

def visualize_normalized_data(x, y, z):
    for index, subarray in enumerate(x):
        run = wandb.init(project='custom-charts', name=name[index])
        x_value = x_range[1] - x_range[0]
        table_x = wandb.Table(data=[[(i / len(subarray)) * x_value + x_range[0], value] for i, value in enumerate(subarray)], columns=["Index", "Value"])
        histogram_x = wandb.plot.histogram(table_x, value='Value', title='X Histogram')

        y_value = y_range[1] - y_range[0]
        table_y = wandb.Table(data=[[(i / len(y[index])) * y_value + y_range[0], value] for i, value in enumerate(y[index])], columns=["Index", "Value"])
        histogram_y = wandb.plot.histogram(table_y, value='Value', title='Y Histogram')

        z_value = z_range[1] - z_range[0]
        table_z = wandb.Table(data=[[(i / len(z[index])) * z_value + z_range[0], value] for i, value in enumerate(z[index])], columns=["Index", "Value"])
        histogram_z = wandb.plot.histogram(table_z, value='Value', title='Z Histogram')

        wandb.log({
            'histogram_x': histogram_x,
            'histogram_y': histogram_y,
            'histogram_z': histogram_z
        })

        # 结束 W&B 运行
        run.finish()

if __name__ == "__main__":
	normal_x = normal_list(x)
	normal_y = normal_list(y)
	normal_z = normal_list(z)

	visualize_normalized_data(normal_x, normal_y, normal_z)

