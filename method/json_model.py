import json
import numpy as np

def json_to_numpy_for_key(file_path, key_name):
    """
    從 JSON 檔案中讀取指定鍵值的二維陣列並轉換為 NumPy 陣列。
    
    :param file_path: JSON 檔案路徑
    :param key_name: JSON 內的鍵名 (例如 'wall_map')
    :return: 轉換後的 NumPy 2D 陣列
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        if key_name in data:
            # 轉換為 NumPy 陣列，並根據需求指定 dtype (例如 np.int8)
            matrix = np.array(data[key_name], dtype=np.int8)
            return matrix
        else:
            print(f"錯誤：找不到鍵名 '{key_name}'")
            return None
            
    except Exception as e:
        print(f"讀取檔案時發生錯誤: {e}")
        return None
def json_to_numpy(file_path):
    """
    從 JSON 檔案中讀取指定鍵值的二維陣列並轉換為 NumPy 陣列。
    
    :param file_path: JSON 檔案路徑
    :param key_name: JSON 內的鍵名 (例如 'wall_map')
    :return: 轉換後的 NumPy 2D 陣列
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            return data
            
    except Exception as e:
        print(f"讀取檔案時發生錯誤: {e}")
        return None
def update_multiple_out(file_path, best_AP, average_RSSI, average_arr, pref_arr):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    data["best_AP"] = best_AP.tolist() if hasattr(best_AP, 'tolist') else best_AP
    data["average_RSSI"] = float(average_RSSI) # 確保是原生 float
    data["AP_array"] = average_arr.tolist() if hasattr(average_arr, 'tolist') else average_arr
    data["pref_arr"] = pref_arr.tolist() if hasattr(pref_arr, 'tolist') else pref_arr
    json_string = json.dumps(data, separators=(',', ':'))
    
    # 在矩陣的每一列結尾 `],` 後面加上換行符 \n
    formatted_string = json_string.replace('],', '],\n')
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(formatted_string)
    

    print("✅ JSON 資料已成功更新！")
def update_bestAP(file_path, best_AP, average_RSSI, average_rssi_arr, AP_array):


    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    data["best_AP"] = best_AP.tolist() if hasattr(best_AP, 'tolist') else best_AP
    data["average_RSSI"] = float(average_RSSI) # 確保是原生 float
    data["AP_array"] = AP_array.tolist() if hasattr(AP_array, 'tolist') else AP_array
    data["average_rssi_arr"] = average_rssi_arr.tolist() if hasattr(average_rssi_arr, 'tolist') else average_rssi_arr
    json_string = json.dumps(data, separators=(',', ':'))
    
    # 在矩陣的每一列結尾 `],` 後面加上換行符 \n
    formatted_string = json_string.replace('],', '],\n')
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(formatted_string)
    

    print("✅ JSON 資料已成功更新！")
# ==============================
# 使用範例
# ==============================
if __name__ == "__main__":
    file_name = "new_map/map_0000.json"

    # 取得 wall_map 的 NumPy 陣列
    wall_np = json_to_numpy_for_key(file_name, "wall_map")

    if wall_np is not None:
        print(f"成功轉換！陣列形狀: {wall_np.shape}")
        print(f"前 5x5 的內容:\n{wall_np[:5, :5]}")