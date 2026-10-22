# 2026-10-22 專案結構重整

## 背景

這是 2022 碩論時期的研究程式，原本是從 [BirdCLEF-Baseline](https://github.com/kahst/BirdCLEF-Baseline)（Lasagne 版）改寫成 Keras。
改寫過程留下很多舊版殘骸，加上 README 是事後補寫的，導致「文件說的」和「程式實際做的」不一致。
這次整理的目標：**結構清楚、文件誠實、能在任何作業系統重現**，但**不改變模型架構與輸入格式**（Android app 仍要能直接使用匯出的 TFLite）。

## 發現的問題

### 1. class weight 套到錯的類別（實際 bug）
- `model_details.class_weights()` 依 `train_ds.file_paths` 的順序統計每類數量。
  `image_dataset_from_directory(shuffle=True)` 的 `file_paths` 是**打亂後**的順序，所以 dict 的鍵順序是隨機的。
- `create_class_weight()` 把 dict 第 i 個鍵當成 label index `i`，但 Keras 的 label index 是依 `class_names` 字母排序。
  兩者不一致 → 權重被套到錯的鳥種。
- 另外用 `'\\'` 切路徑取類別名，非 Windows 系統上每個檔案都被當成獨立類別。
- **影響**：2022 年論文中的模型是用錯位的 class weight 訓練的。

### 2. README 與程式行為不符
| README 原本寫 | 程式實際行為 |
|---|---|
| `config.py` 是唯一設定來源 | 訓練腳本完全沒 import `config`，參數寫死在程式裡 |
| Adam + cosine LR schedule | Nadam + `ReduceLROnPlateau` |
| 60 epochs、early stopping 10 | 迴圈 99 次 × `fit(epochs=1)`，沒有 early stopping |
| roll / noise 資料增強 | 訓練時沒有任何增強 |
| 輸出到 `snapshots/`、`BirdCLEF_Logfile.txt` | 實際輸出到 `my-model-save/resnet/<name>/` |
| 會爬 xeno-canto、有 EfficientNet 版本 | repo 中沒有這些程式 |

### 3. `ReduceLROnPlateau` 從未生效
每次 `fit` 只跑 1 個 epoch，callback 狀態每輪重置，`patience=5` 永遠達不到。

### 4. 只能在 Windows 跑
- 寫死 `"\\?\"` 長路徑前綴、`'error_message\...'` 反斜線路徑。
- `distutils`（Python 3.12 已移除）、`keras.utils.vis_utils`（新版 Keras 已移除）。

### 5. 版控雜物與死碼
- `__pycache__/*.pyc`、`*.py.bak` 被 commit，沒有 `.gitignore`、沒有 `requirements.txt`。
- 未使用的 Lasagne 殘留：`utils/batch_generator.py`、`stats.py`、`metrics.py`、`image.py`。
- `model_details.py` 中未使用的函式：`class_weights2`、`class_weights_v2`、`save_model_v2`、`preprocess_image` 等。
- IDE 自動補的無用 import：`cProfile.label`、`ast.Param`、`tabnanny.check`、`typing_extensions.runtime`。
- `config.py` 大半是 BirdCLEF 測試 / 增強 / Lasagne 模型的設定，沒有任何程式讀取。

## 做了什麼

### 結構
```
舊                                          新
spec.py                                  →  extract_specs.py
dataset-download/mp3_to_wav.py           →  convert_mp3.py（改用命令列參數）
mytrain_resnet18_ver1.py                 →  train.py（流程） + model.py（架構）
model_details.py                         →  utils/training.py（只留用得到的）
dataset-download/dataset_input_folder_name/   →  data/audio/
dataset-download/dataset_output_folder_name/  →  data/specs/
dataset-download/dataset_output_folder_name_new_noise/ → data/noise/
my-model-save/resnet/resnet_0919_test/   →  models/resnet_0919_test/
```
原本提案是把前處理放到 `preprocess/` 子資料夾，最後改成放根目錄：子資料夾內的腳本要用 `python -m` 執行才找得到 `config`，對一個研究 repo 來說多此一舉。

### 行為修正
- **class weight**：改用 `os.path` 取類別名，並依 `class_names` 對應 index，跟 `file_paths` 順序無關。
- **訓練流程**：改成單次 `model.fit(epochs=EPOCHS, initial_epoch=...)`。
  - 續訓：自動從 `checkpoints/` 找最新的 epoch 繼續。
  - 紀錄：`CSVLogger` 取代自製的 pkl 搬移 / csv 寫入。
  - `ReduceLROnPlateau` 因此能正常生效。
- **設定**：`train.py` 改讀 `config.py`；`config.py` 的值設成**原本訓練腳本實際使用的值**（Nadam、val split 0.2、`FILTERS=[16,32,64,128]`、`RESNET_K=2`、`RESNET_N=2`…）。
- **Early stopping**：新增 `EARLY_STOPPING_PATIENCE` 設定，預設 `None`（關閉），與原本行為一致。
- **路徑**：全部改成 `os.path`，移除 Windows 專用寫法。若 Windows 上遇到長路徑問題，請開啟系統的 `LongPathsEnabled`。
- **`utils/audio.py` 小修**：`melspec` 中 `mlspec` 拼字錯誤（全零頻譜時寫錯變數，實際無影響）；`stft` 的 `window='hamm'` 改成 `'hamming'`（新版 scipy 已移除別名）；`signal2noise` 用 `spec.size` 取代 try/except 計算面積（結果相同）。
- **`convert_mp3.py`**：改用命令列參數指定資料夾，移除沒被呼叫的 `remove_bird_folder()`。
- **相依套件**：更新到目前的 TensorFlow / Keras 3。checkpoint 由 SavedModel 資料夾（`.ckpt/`）改成 `.keras` 檔（Keras 3 規定的格式）。

### 刻意保留不變
- 模型架構逐層相同（已比對新舊版的層設定與參數量）。
- 輸入格式：`128×256×1` 灰階、像素值 **0–255、不做 rescale**。Android app 端就是餵 0–255，若改成正規化 app 會壞掉。
- 頻譜產生參數（44.1 kHz、1 秒切片、0.25 秒 overlap、500–15000 Hz mel）。
- 雜訊片段仍會存到 `data/noise/`（原設計用於增強），但目前訓練沒有用到，README 已註明。

## 驗證方式（TensorFlow 2.21 / Keras 3.15 / Python 3.13）
- **模型架構**：用舊版 `resnet_18()` 與新版 `build_resnet()` 各建一個模型，59 層的設定逐層相同、參數量都是 2,796,355。
- **頻譜**：用 repo 內的範例音檔，舊版 `spec.py` 與新版 `extract_specs.py` 產生的 334 張 PNG 逐位元相同。
- **訓練流程**：範例資料跑 2 個 epoch → 改 `EPOCHS=3` 再執行，確認從 `weights.02.keras` 續訓、`training_log.csv` 接續、TFLite 輸入 `[1,128,256,1]` 與參考模型一致。
- **class weight bug 重現**：類別 A/B/C 各 10/1000/50 張並打亂。舊程式的 dict 順序變成 `B, C, A`，最稀少的 A（index 0）拿到權重 1.0；新程式給 A 2.77、B 1.0、C 1.16。

## Code review 與對抗性驗證後的修正

PR 開出後補做了 code review 和對抗性驗證（另開一個獨立驗證者，不給它結論，只要求它想辦法推翻 PR 的每一項宣稱）。

### 修正
| 來源 | 問題 | 修正 |
|---|---|---|
| review | 預設 `RUN_NAME='resnet_0919_test'`，一跑訓練就會蓋掉 commit 進來的參考模型 | 預設改成 `resnet_run` |
| review + 對抗 | 續訓時資料集的類別若有增減／改名，checkpoint 的輸出會靜默對到錯的鳥種（和 class weight bug 同一類問題） | 續訓前比對輸出數量和已存的 `*_labels.txt`，不一致就停止並提示換 `RUN_NAME` |
| review | `CSVLogger` 不管是不是新的訓練都用 append，會混到舊紀錄 | 只有續訓時才 append |
| review | 訓練集裡沒有樣本的類別拿到最小權重 1.0 | 當作 1 個樣本計算，因此拿到最大權重 |
| review | `tensorflow>=2.16` 的早期版本無法把 Keras 3 模型轉成 TFLite | 下限改成實測過的 2.21 |
| review | 每個 epoch 直接覆寫 tflite，中斷時可能留下壞檔 | 先寫暫存檔，再用 `os.replace` |
| review | 自訂 `FILTERS` 讓 stride 2 時通道數不變的話，shortcut 形狀會對不上 | stride≠1 時也做投影（預設設定的架構不變，已重新比對） |
| review | 輸入資料夾根目錄的 mp3 會轉到 `data/audio/` 根目錄，然後被忽略 | 略過並提示放進物種子資料夾 |
| review | 轉檔失敗會留下不完整的 wav，下次也不會重試 | 失敗時刪掉 |
| 對抗 | **Python 3.13 移除了 `audioop`，pydub 一 import 就失敗** | requirements 加上 `audioop-lts` |
| 對抗 | 輸入路徑不存在時仍印出 done、回傳成功 | 改成報錯 |
| 對抗 | 舊版 `use_gpu()` 會設定 `CUDA_VISIBLE_DEVICES='0'`，新版漏掉了 | 加回來，改用 `setdefault`，可以從外部覆寫 |

### 判定不修
- checkpoint 檔名的 regex 和 `CHECKPOINT_PATTERN` 各寫一份；最新的 checkpoint 損壞時不會退回前一個 → 很少遇到，加了會增加複雜度。
- 轉檔時按 Ctrl-C 留下的不完整 wav、錯誤 log 重複寫入 → 舊版也一樣，影響很小。
- TFLite 轉換時會印出大量 `TensorSpec` 訊息 → 只是輸出太多，不影響功能。
- `data/specs/` 內附的 3 張範例圖來自不在 `data/audio/` 的錄音 → 只是範例，執行 `extract_specs.py` 後就會被正常資料蓋過去。

### 對抗性驗證確認成立的宣稱
模型架構逐層相同（n=3 和 n=610 都比對過，包含拓撲）、頻譜 PNG 逐位元相同、舊版 class weight 確實錯位、TFLite I/O 和 op 數量與參考模型一致、超參數和舊訓練腳本一致、`ReduceLROnPlateau` 會實際觸發（0.001 → 0.0003）。

## 未來可以做的（這次沒做）
- 把資料增強真的接進訓練（`data/noise/` 混音、vertical roll）。
- 用修正後的 class weight 重新訓練，和 `models/resnet_0919_test` 比較準確率。
- 加測試 / CI。
