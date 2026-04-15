from typing import List
import numpy as np

def binary_to_proportion(binary_population: List[str]) -> np.ndarray:


    element_component = []
    for binary_str in binary_population:
        if len(binary_str) != 23:
            raise ValueError(f"二进制字符串长度错误：{len(binary_str)}，应为23位")


        E1 = int(binary_str[0])*8 + int(binary_str[1])*4 + int(binary_str[2])*2 + int(binary_str[3])*1
        E2 = int(binary_str[4])*8 + int(binary_str[5])*4 + int(binary_str[6])*2 + int(binary_str[7])*1
        E3 = int(binary_str[8])*16 + int(binary_str[9])*8 + int(binary_str[10])*4 + int(binary_str[11])*2 + int(binary_str[12])*1
        E4 = int(binary_str[13])*16 + int(binary_str[14])*8 + int(binary_str[15])*4 + int(binary_str[16])*2 + int(binary_str[17])*1
        E5 = int(binary_str[18])*16 + int(binary_str[19])*8 + int(binary_str[20])*4 + int(binary_str[21])*2 + int(binary_str[22])*1

        element_component.append([E1, E2, E3, E4, E5])


    element_array = np.array(element_component, dtype=np.float32)


    proportion_array = []
    for row in element_array:
        total = np.sum(row)

        proportion = row / total


        proportion_rounded = np.round(proportion, decimals=2)

        delta = 1.0 - np.sum(proportion_rounded)

        max_index = np.argmax(proportion_rounded)

        proportion_rounded[max_index]+=delta

        proportion_rounded = np.around(proportion_rounded, decimals=2)

        proportion_array.append(proportion_rounded)

    return np.array(proportion_array)