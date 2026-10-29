import os
# ============================================================
# 程式功能:使用cupy套件，利用gpu做多線程的高速RSSI heatmap計算
#!!!!!!!!!!!!!!非常重要!!!!!!!!!!!!!!!
#如因為環境問題導致cupy設定失敗，無法使用gpu，請把cupy改成numpy(使用cpu運行)，不影響結果
# ============================================================

# 1. 指向 14.30 的路徑
msvc_base = r'C:\Program Files\Microsoft Visual Studio\18\Insiders\VC\Tools\MSVC\14.30.30705'
# 2. 指向 Windows SDK 的路徑 (請確認你的版本號，例如 10.0.22621.0)
sdk_base = r'C:\Program Files (x86)\Windows Kits\10\Include\10.0.22621.0'

# 加入標頭檔搜尋路徑 (解決 crtdefs.h 找不到的問題)
include_paths = [
    os.path.join(msvc_base, 'include'),
    os.path.join(sdk_base, 'ucrt'),
    os.path.join(sdk_base, 'shared'),
    os.path.join(sdk_base, 'um'),
]

# 將路徑注入 INCLUDE 環境變數
os.environ['INCLUDE'] = os.pathsep.join(include_paths) + os.pathsep + os.environ.get('INCLUDE', '')

# 將 cl.exe 加入 PATH
os.environ['PATH'] = os.path.join(msvc_base, r'bin\Hostx64\x64') + os.pathsep + os.environ['PATH']

print('Configured SDK and MSVC include paths for CuPy propagation')
import cupy as cp
import config
import config_radio as radio


class Cal_db_gpu:
    def __init__(self, tx_power=None, walls=None, frequency_ghz=None,
                 grid_size=None, ray_steps=None, row_chunk=None):
        self.tx_power = radio.TX_POWER_DBM if tx_power is None else tx_power
        self.frequency_ghz = radio.ACCESS_FREQUENCY_GHZ if frequency_ghz is None else frequency_ghz
        self.grid_size = radio.GRID_SIZE_METERS if grid_size is None else grid_size
        self.ray_steps = config.RSSI_RAY_STEPS if ray_steps is None else ray_steps
        self.row_chunk = config.RSSI_ROW_CHUNK if row_chunk is None else row_chunk
        self.walls = cp.asarray(walls, dtype=cp.float32) if walls is not None else cp.zeros((10,10),cp.float32)

    def _path_attenuation(self, ax, ay, gx, gy, dist):
        """精確計算射線在每個格子內的穿越長度；wall_map 的原始值直接當 dB/格。"""
        h, w = self.walls.shape
        # 座標表示格子中心 (x, y)，格子邊界位於整數座標 +/- 0.5。
        dx = gx - ax
        dy = gy - ay
        dx_safe = cp.where(dx == 0, 1, dx)
        dy_safe = cp.where(dy == 0, 1, dy)

        # 射線與所有垂直、水平格線的交點參數 t (0~1)。
        vertical = cp.arange(w - 1, dtype=cp.float32)[:, None, None] + 0.5
        horizontal = cp.arange(h - 1, dtype=cp.float32)[:, None, None] + 0.5
        tx = (vertical - ax) / dx_safe[None, :, :]
        ty = (horizontal - ay) / dy_safe[None, :, :]
        tx = cp.where((dx[None, :, :] != 0) & (tx > 0) & (tx < 1), tx, 1)
        ty = cp.where((dy[None, :, :] != 0) & (ty > 0) & (ty < 1), ty, 1)

        zeros = cp.zeros((1,) + gx.shape, dtype=cp.float32)
        ones = cp.ones((1,) + gx.shape, dtype=cp.float32)
        t = cp.sort(cp.concatenate((zeros, tx, ty, ones), axis=0), axis=0)
        # 相鄰交點間的中點必定落在單一格子內，直接乘上該段的實際長度。
        dt = t[1:] - t[:-1]
        midpoint = (t[1:] + t[:-1]) * 0.5
        ix = cp.clip(cp.floor(ax + dx[None, :, :] * midpoint + 0.5).astype(cp.int32), 0, w - 1)
        iy = cp.clip(cp.floor(ay + dy[None, :, :] * midpoint + 0.5).astype(cp.int32), 0, h - 1)
        # dist 單位為格；原始 walls 值代表每完整穿越一格的衰減 dB。
        return cp.sum(self.walls[iy, ix] * dt, axis=0) * dist

    def calculate_signal_map(self, ap_pos):
        h,w = self.walls.shape
        ax,ay = ap_pos
        result = cp.empty((h,w), dtype=cp.float32)
        frequency_mhz = self.frequency_ghz*1000.
        t = cp.linspace(0,1,self.ray_steps,dtype=cp.float32)[:,None,None]
        for start in range(0,h,self.row_chunk):
            stop=min(start+self.row_chunk,h)
            gy,gx=cp.meshgrid(cp.arange(start,stop),cp.arange(w),indexing='ij')
            dist=cp.sqrt((gx-ax)**2+(gy-ay)**2)
            meters=cp.maximum(dist*self.grid_size,0.5)
            fspl=32.44+20*cp.log10(frequency_mhz)+20*cp.log10(meters/1000.)
            attenuation=self._path_attenuation(ax,ay,gx,gy,dist)
            rssi=cp.clip(self.tx_power-fspl-attenuation,-100.,-20.)
            result[start:stop]=cp.where(self.walls[start:stop]>0,cp.nan,cp.round(rssi,2))
        valid=cp.isfinite(result)
        average=cp.mean(result[valid]) if bool(cp.any(valid)) else -100.
        return result,average
