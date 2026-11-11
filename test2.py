import math

def milliwatts_to_dbm(milliwatts):
    """將功率（毫瓦 mW）轉換為 dBm"""
    if milliwatts <= 0:
        raise ValueError("功率必須大於 0（對數運算限制）")
    return 10 * math.log10(milliwatts)

mw_input = 1.4162e-7
dbm_result1 = milliwatts_to_dbm(mw_input)
print(f"{mw_input} mW = {dbm_result1:.2f} dBm")  # 預期：0.00 dBm