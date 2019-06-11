file_predict = open("result_log.txt", mode="r")
file_label = open("label_processed_digits.txt", mode="r")

predicts = file_predict.readlines()
label = file_label.readlines()
error = 0
for i in range(len(predicts)):

	if label[i].split(", ")[1].replace("\n", "") != predicts[i].split(", ")[1].replace("\n", ""):
		error += 1

print(1 - (error/len(predicts)))
	