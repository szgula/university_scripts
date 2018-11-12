import numpy as np
import matplotlib.pyplot as plt

corelation_list = []

file_list = ["55cm.lvm", "74cm.lvm", "85cm.lvm", "105cm.lvm", "116cm.lvm", "123cm.lvm", "141cm.lvm"]
translation_list = []
distance_list = []

dt = 1.041667E-5

def autocorr(x_1, x_2):
    result = np.correlate(x_1, x_2, mode='full')
    return result

for file in file_list:
    x_1 = []
    x_2 = []
    with open(file) as f:
        for line in f:
            values = line.replace(',', '.')
            values = values.replace('\n', '')
            values = values.split("\t")
            if len(values) == 3:
                x_1.append(float(values[1]))
                x_2.append(float(values[2]))
            else:
                break
    result = autocorr(x_1, x_2)
    np_result = np.array(result)
    trenslation = np.argmax(np_result)
    trenslation -= (len(np_result) / 2)

    translation_list.append(trenslation)
    distance = file.replace('cm.lvm', '')
    distance = int(distance)
    distance_list.append(distance)

print ("przesuniecie sygnalow (ilosc probek): ", translation_list)
distance_list = np.array(distance_list)
distance_list = np.divide(distance_list, 100.0)
print ("odleglosc mikrofonow (m): ", distance_list)

delta_time_between_microphones = np.array(translation_list)
delta_time_between_microphones *= dt

print("przesuniecie sygnalow (s): ", delta_time_between_microphones)

speed_of_sound = np.divide(distance_list, delta_time_between_microphones)

print ("predkosc dzwieku (m/s): ", abs(speed_of_sound))

#355.55544178, 302.94233376, 354.0129019,  329.95079458, 332.91469167, 353.00437134, 349.31601725