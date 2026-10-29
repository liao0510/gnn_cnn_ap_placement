"""隨機生成矩形房間平面圖；直接修改下方參數後執行。"""
import json
import random
import re
from collections import deque
from pathlib import Path

import numpy as np

# ==================== 使用者設定 ====================
MAP_WIDTH = 1000             # 地圖寬度（格）
MAP_HEIGHT = 1000            # 地圖高度（格）
NUM_SAMPLES = 10           # 生成地圖數量
WALL_THICKNESS = 2          # 牆壁厚度（格）：外牆與內牆皆適用
NUM_FORBIDDEN = 5           # 每張地圖的避免區數量
JSON_SAVE_DIR = "./maps_1000_1000"     # JSON 儲存資料夾
JSON_FILE_NAME = "map"     # 檔名前綴，例如 house_0001.json
RANDOM_SEED = 42            # None：每次生成不同結果

MIN_ROOM_SIZE = 12          # 每個房間最小淨寬與淨高（格）
MAX_SPLIT_DEPTH = 1000         # 最多分割層數
DOOR_WIDTH = 5              # 門洞寬度（格）
FORBIDDEN_MIN_SIZE = 2      # 避免區最小邊長（格）
FORBIDDEN_MAX_SIZE = 6      # 避免區最大邊長（格）
MATERIALS = (1, 3, 6, 10)      # 木牆、磚牆、水泥牆：示意衰減值（dB）
MATERIAL_WEIGHTS = (1, 3, 3, 3)
EXTERIOR_MATERIAL = 10
# ====================================================


def floor_is_connected(walls):
    """四方向 BFS：確保所有可通行格子互相連通。"""
    floor = walls == 0
    points = np.argwhere(floor)
    if len(points) == 0:
        return False
    h, w = floor.shape
    sy, sx = map(int, points[0])
    visited = np.zeros_like(floor, dtype=bool)
    visited[sy, sx] = True
    queue = deque([(sy, sx)])
    count = 0
    while queue:
        y, x = queue.popleft()
        count += 1
        for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
            if 0 <= ny < h and 0 <= nx < w and floor[ny, nx] and not visited[ny, nx]:
                visited[ny, nx] = True
                queue.append((ny, nx))
    return count == int(floor.sum())


def generate_floorplan(width, height, rng, wall_thickness=2,
                       num_forbidden=3, min_room_size=12, max_depth=5,
                       door_width=3):
    """用 BSP 切割矩形房間，牆壁為指定厚度，隔間留有門洞。"""
    t = wall_thickness
    if not isinstance(t, int) or t < 1:
        raise ValueError("WALL_THICKNESS 必須是正整數")
    if width < 2 * t + min_room_size or height < 2 * t + min_room_size:
        raise ValueError("地圖太小，無法容納外牆與最小房間")
    if min_room_size < 4:
        raise ValueError("MIN_ROOM_SIZE 至少為 4")
    if not 1 <= door_width <= min_room_size - 2:
        raise ValueError("DOOR_WIDTH 必須介於 1 與 MIN_ROOM_SIZE-2")
    if num_forbidden < 0:
        raise ValueError("NUM_FORBIDDEN 不可小於 0")

    walls = np.zeros((height, width), dtype=np.uint8)
    walls[:t, :] = EXTERIOR_MATERIAL
    walls[-t:, :] = EXTERIOR_MATERIAL
    walls[:, :t] = EXTERIOR_MATERIAL
    walls[:, -t:] = EXTERIOR_MATERIAL
    rooms = []

    def split(x0, y0, x1, y1, depth):
        """x1/y1 為包含在內的右/下邊界。"""
        rw, rh = x1 - x0 + 1, y1 - y0 + 1
        axes = []
        if rw >= 2 * min_room_size + t:
            axes.append("vertical")
        if rh >= 2 * min_room_size + t:
            axes.append("horizontal")
        if not axes or depth >= max_depth or (depth > 0 and rng.random() < 0.17):
            rooms.append((x0, y0, x1, y1))
            return

        if len(axes) == 2:
            axis = ("vertical" if rw / rh > rng.uniform(0.85, 1.15)
                    else "horizontal")
        else:
            axis = axes[0]
        material = rng.choices(MATERIALS, weights=MATERIAL_WEIGHTS, k=1)[0]

        if axis == "vertical":
            cut = rng.randint(x0 + min_room_size, x1 - min_room_size - t + 1)
            walls[y0:y1 + 1, cut:cut + t] = material
            gap = rng.randint(y0 + 1, y1 - door_width)
            walls[gap:gap + door_width, cut:cut + t] = 0
            split(x0, y0, cut - 1, y1, depth + 1)
            split(cut + t, y0, x1, y1, depth + 1)
        else:
            cut = rng.randint(y0 + min_room_size, y1 - min_room_size - t + 1)
            walls[cut:cut + t, x0:x1 + 1] = material
            gap = rng.randint(x0 + 1, x1 - door_width)
            walls[cut:cut + t, gap:gap + door_width] = 0
            split(x0, y0, x1, cut - 1, depth + 1)
            split(x0, cut + t, x1, y1, depth + 1)

    split(t, t, width - t - 1, height - t - 1, 0)

    # 每個避免區都是一塊矩形，只能放在無牆的室內位置。
    forbidden = np.zeros_like(walls, dtype=np.uint8)
    if num_forbidden:
        candidates = rooms.copy()
        rng.shuffle(candidates)
        placed = 0
        attempts = 0
        while placed < num_forbidden and attempts < max(200, num_forbidden * 200):
            attempts += 1
            x0, y0, x1, y1 = rng.choice(candidates)
            available_w, available_h = x1 - x0 - 1, y1 - y0 - 1
            if available_w < FORBIDDEN_MIN_SIZE or available_h < FORBIDDEN_MIN_SIZE:
                continue
            fw = rng.randint(FORBIDDEN_MIN_SIZE, min(FORBIDDEN_MAX_SIZE, available_w))
            fh = rng.randint(FORBIDDEN_MIN_SIZE, min(FORBIDDEN_MAX_SIZE, available_h))
            fx = rng.randint(x0 + 1, x1 - fw)
            fy = rng.randint(y0 + 1, y1 - fh)
            area = forbidden[fy:fy + fh, fx:fx + fw]
            if np.any(walls[fy:fy + fh, fx:fx + fw]) or np.any(area):
                continue
            area[:] = 1
            placed += 1
        if placed < num_forbidden:
            raise RuntimeError(
                f"只能放入 {placed}/{num_forbidden} 個避免區，"
                "請降低 NUM_FORBIDDEN 或縮小避免區尺寸"
            )
    return walls, forbidden


def generate_dataset(width, height, count, wall_thickness, num_forbidden,
                     output_dir=None, file_name="house", seed=None, min_room_size=12,
                     max_depth=5, door_width=3):
    if count < 1:
        raise ValueError("NUM_SAMPLES 必須大於 0")
    output = Path(output_dir or f"map_{width}_{height}")
    output.mkdir(parents=True, exist_ok=True)
    if not file_name or file_name in (".", "..") or "/" in file_name or "\\" in file_name:
        raise ValueError("JSON_FILE_NAME 必須是不含路徑符號的檔名前綴")
    # 只計算相同前綴的 JSON；使用最大編號，避免刪掉中間檔案後覆蓋既有資料。
    filename_pattern = re.compile(rf"^{re.escape(file_name)}_(\d+)\.json$")
    existing_ids = [int(match.group(1)) for item in output.iterdir()
                    if item.is_file() and (match := filename_pattern.fullmatch(item.name))]
    start_id = max(existing_ids, default=-1) + 1
    # 固定 seed 時，續接批次也不會重複從第一張地圖開始。
    rng = random.Random(seed)
    if seed is not None:
        rng = random.Random(f"{seed}:{start_id}")
    print(f"起始編號：{start_id}；儲存位置：{output.resolve()}")
    created = []
    for index in range(count):
        sample_id = start_id + index
        for attempt in range(100):
            walls, forbidden = generate_floorplan(
                width, height, rng, wall_thickness, num_forbidden,
                min_room_size, max_depth, door_width
            )
            if floor_is_connected(walls):
                break
        else:
            raise RuntimeError("連續 100 次都無法產生完全連通的地圖")

        data = {
            "id": sample_id,
            "width": width,
            "height": height,
            "wall_map": walls.tolist(),
            "forbidden": forbidden.tolist(),
        }
        target = output / f"{file_name}_{sample_id:04d}.json"
        save_as_formatted_json(data, target)
        # with target.open("x", encoding="utf-8") as f:
        #     json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
        created.append(target)
        if (index + 1) % 50 == 0 or index + 1 == count:
            print(f"已完成 {index + 1}/{count}；最新檔案：{target.name}")
    return created
def save_as_formatted_json(data, filename):
    # 將資料轉為無空白的 JSON 字串
    json_string = json.dumps(data, separators=(',', ':'))
    
    # 在矩陣的每一列結尾 `],` 後面加上換行符 \n，這樣肉眼看就像矩陣
    formatted_string = json_string.replace('],', '],\n')
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(formatted_string)

if __name__ == "__main__":
    files = generate_dataset(
        width=MAP_WIDTH,
        height=MAP_HEIGHT,
        count=NUM_SAMPLES,
        wall_thickness=WALL_THICKNESS,
        num_forbidden=NUM_FORBIDDEN,
        output_dir=JSON_SAVE_DIR,
        file_name=JSON_FILE_NAME,
        seed=RANDOM_SEED,
        min_room_size=MIN_ROOM_SIZE,
        max_depth=MAX_SPLIT_DEPTH,
        door_width=DOOR_WIDTH,
    )
    print(f"地圖已儲存至：{files[0].parent.resolve()}")
