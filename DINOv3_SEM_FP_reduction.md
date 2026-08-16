# DINOv3 SEM 缺陷偵測：False Positive 問題診斷與改善方案

> 給 AI coding agent 的工作說明。目標：在 image-level 維持 TP（Recall ≥ 0.93）的前提下，顯著降低 FP。

---

## 1. 專案背景

### 1.1 資料

| 用途 | 內容 | 數量 |
|---|---|---|
| SSL pretrain | domain-specific SEM images（無標註） | 1M |
| Few-shot train / val | 含 bounding box 標註（**全部都是含缺陷的影像**） | 50 / 50 |
| BBox test | 含 bounding box 標註（**全部都是含缺陷的影像**） | 1,436 |
| 實際部署 test | 僅 image-level label（defect / normal） | 30K |

- 只有**單一類別**：判斷 defect / normal。
- 30K 的 defect rate：**2~6%**。
- 30K 的 GT 產生方式：線上 1000-YOLOv26 初步篩選 → 人工複判。
- SEM 影像為**灰階**。

### 1.2 模型配置

**Model A — DINOv3**
- Backbone: DINOv3 ViT-B/**16**，含 1M domain images SSL pretrain
- Neck: SimpleFPN (p2~p4)
- Decoder: **YOLOv26 e2e decoder（含 one-to-one + one-to-many）**
- Fine-tune: layer decay 0.75 + weight decay
- 超參數與 augmentation recipe **完全沿用 Ultralytics YOLOv26 official**

**Model B — YOLOv26**
- B1: 50 張 from scratch
- B2: >1000 張 from scratch（線上服役中，只能取得 1/0 prediction，無 confidence score）

### 1.3 目前結果

**BBox test（1,436 張，全部含缺陷）**
```
R & P & AUPRC & R@90P:  DINO-SSL > DINO-FT > YOLOv26-50FS
```

**實際 30K test（image-level）**
```
P:      YOLO-1000 > YOLO-50FS  > DINO-SSL  > DINO-50FT
R:      YOLO-1000 > DINO-SSL   > DINO-50FT > YOLO-50FS
AUPRC:  DINO-SSL  > DINO-50FT  > YOLO-50FS
R@90P:  YOLO-50FS > DINO-SSL   > DINO-50FT
```

**核心症狀**
- DINO 的 Recall 不錯（>0.93），接近 YOLO-1000
- DINO 的 **Precision 與 R@90P 落後可達 >10%**
- DINO 的 **FP 明顯多於 YOLO**
- DINO 的 **bounding box 定位反而比 YOLO 準**

**已做過的嘗試**
- 把 mosaic + HSV + EMA 加進 DINO fine-tune（沿用 official recipe，`close_mosaic=10`，總 350 epochs）
- 結果（皆在 **bbox test** 上）：
  - **bbox 定位變弱很多**
  - R / P / AUPRC / R@90P **些微優於 YOLOv26-50FS，但全部弱於 DINO-50FT**
  - 即：**mosaic 在 bbox test 上全面劣化了 DINO**，只是仍未跌破 YOLOv26-50FS
- image-level 指標**尚未測試**

---

## 2. 問題診斷

### 2.1 關鍵訊號：AUPRC 贏但 R@90P 輸

這兩個指標方向相反，意義明確：

> DINO 的 PR 曲線整體較好，但**在高 precision 端存在一條「高信心 FP 的尾巴」** —— 有一群 normal 影像被打了很高的分數。

問題**不是**排序能力差、**不是**特徵品質差，而是**特定一群 normal 影像被過度自信地誤判**。

以 defect rate 4% 估算，10% 的 precision 差距約對應到 **150 張左右的特定 normal 影像**。這是可以逐張檢視的量級，失效模式幾乎必然集中在少數幾個 cluster，而非均勻雜訊。

---

### 2.2 根因 A：Patch-16 在小缺陷上沒有解析度餘裕（主因）

ViT/16 在**第一層 patch projection 就把 16×16 區塊內的高頻資訊平均掉**。

SimpleFPN 的 p2 名義上是 stride-4，但資訊源頭仍是 stride-16 的 token map —— **它是插值捏出來的「假高解析度」，資訊量並未增加**。

這造成兩種能力的分裂：

| 任務 | 性質 | 在粗特徵下的表現 |
|---|---|---|
| BBox 回歸 | 平滑問題，插值可還原 | **好** ← 解釋「bbox 抓得比較準」 |
| Defect / nuisance 判別 | 需要局部高頻對比、邊緣銳度、紋理相位 | **差** ← 解釋「FP 多」 |

對照組 YOLOv26 的 CNN stem 保留 stride-2/4 的高頻通路，在判別任務上天生有優勢。

**這一條根因單獨就能解釋全部觀察結果。**

---

### 2.3 根因 B：Global attention 學到全域捷徑

50 張 few-shot 訓練影像**全部都是含缺陷的影像**，訓練集中完全沒有 negative image。

ViT 的 global attention 在這種資料下極容易學到捷徑：

> 「這張影像的整體亮度 / 對比 / pattern 統計分布 → 有缺陷」

而非真正定位缺陷本身。

這種捷徑製造的失效模式正好是「**整張 normal 影像被高分誤判**」，與 2.1 觀察到的「高信心 FP 尾巴」完全吻合。

CNN 因為感受野受限，天生不容易學到這種全域捷徑 —— 這解釋了為什麼**同樣 50 張資料、同樣 decoder、同樣超參數**，YOLO 的 image-level precision 反而較好。

> ⚠️ **證據狀態：此為假說，目前尚無直接證據。**
> 原因是唯一能驗證它的 benchmark（bbox test）**完全不含 normal 影像**，結構上量不到這個失效模式（見 2.5-(3)）。
> 驗證方式見 **4.2**（原生解析度 random crop）—— 它能在不損失解析度的前提下單獨隔離此效應。

---

### 2.4 根因 C：Mosaic 全面劣化 bbox 指標（且 image-level 影響仍屬未知）

因為影像是**灰階**，`hsv_h` / `hsv_s` 對三通道複製的灰階影像是 **no-op**，只有 `hsv_v`（亮度增益）實際生效；EMA 對 50 張規模是純益、不會傷定位。

**因此該次實驗的實際變數幾乎就是 mosaic 單獨的效果。**

**實際結果（皆在 bbox test 上）：mosaic 版本在 R / P / AUPRC / R@90P 上全部弱於 DINO-50FT，定位更是明顯變差。** 只是仍些微優於 YOLOv26-50FS。

**即：在 bbox test 這個 benchmark 上，mosaic 對 DINO 沒有任何面向是有益的。**

機制解釋 —— 這與**根因 A** 完全一致：

Ultralytics mosaic 拼 2×imgsz 畫布後裁回 imgsz，加上 `scale=0.5`（0.5×~1.5× 抖動），物件平均呈現尺度約砍半：

- YOLO（stride-4/8）：24px 缺陷 → 12px，P2 上仍有 3×3 cell，**還活著**
- DINO ViT/16：24px 缺陷原本只有 ~1.5 個 token → 砍半後**不到 1 個 token，直接消失在 patch projection 裡**

因此 mosaic 對 patch-16 backbone 是**淨傷害**：它把根因 A 的解析度餘裕吃光，連帶拖垮分類與回歸。

**但必須注意一個 benchmark 盲區（見 2.5-(3)）：**

> bbox test 的 1,436 張**全部含缺陷，沒有任何 normal 影像**。
> 因此這個 benchmark **在結構上就量不到「normal 影像被誤判」這個失效模式** —— 而那正是 30K 上 FP 的主要來源。

意即：**mosaic 是否有打斷根因 B 的全域捷徑（進而降低 image-level FP），在現有證據下無法判斷 —— 不是被否證，是根本沒被測到。** 已知的只有它明確傷害了 bbox test 上的所有指標。

**結論**：不應把 mosaic 當作解法；但由於它同時混合了「傷害解析度」與「可能打斷全域捷徑」兩個相反效應，**應改用能單獨隔離後者、且不付解析度代價的做法（見 4.2）**。

補充：`close_mosaic=10` epochs 在 50 張資料上約等於 60 optimizer steps（vs COCO 上約 18,500 steps）。以總 350 epochs 計，clean 階段佔 ~3%（COCO 為 10%），偏差存在但**非主因** —— 主因是解析度。

---

### 2.5 評估協定的兩個缺陷

**(1) `threshold = 0.5` 對兩個模型不等價**

DETR-style head 使用 focal loss、且僅以 50 張影像訓練，輸出分數尺度與 YOLO 完全不同。0.5 落在兩條 PR 曲線上的位置是任意的 —— **用它比較本身就引入測量誤差**。

**(2) `image_score = max(box_scores)` 過於脆弱**

在 30K 影像 × 數百 query 的規模下，這是極值統計問題：query 數越多，某個 query 在 normal 影像上「爆分」的機率越高。

**(3) BBox test set 量不到目標指標**

1,436 張全部含缺陷，其上的 Precision 測的是「在缺陷圖中框錯位置」；而 30K 上的 FP 主要來自「純 normal 影像被誤判」。**這是兩種不同的失效模式，因此 bbox benchmark 與部署指標的相關性很低。**

---

### 2.6 已知的評估偏誤（無法修正，但需知悉）

30K 的 GT 由「1000-YOLO 篩選 → 人工複判」產生，因此：

- 1000-YOLO 在此 GT 上的 recall **在定義上接近 1.0**，非真實能力
- **1000-YOLO 漏掉的缺陷全部被標為 normal**；DINO 若抓到，會被計為 FP

意即目前 DINO 的 FP 混合了兩種東西：真正的誤判，以及「抓到現行系統漏檢的真缺陷」。此偏誤無法在現有資料下移除，但應在結論中揭露。

---

## 3. 可檢查的方向（除錯，非調參）

### 3.1 Normalization 一致性 【高優先，10 分鐘】

灰階複製成三通道時，使用的 `mean` / `std` 是否與 **1M SSL pretrain 階段完全一致**？

- 若 dataloader 直接套 `/255` 或 ImageNet RGB 統計，而 SSL 使用另一組統計 → backbone 運作在偏移的輸入分布上
- **此類 mismatch 的典型症狀正是「粗結構撐得住、細粒度判別力下降」** —— 與目前症狀（bbox 準、FP 多）完全吻合
- 這是 bug 而非方法問題

**檢查點**：SSL 訓練 config 的 normalization 參數 vs detection dataloader 的 transform pipeline。

---

### 3.2 Patch token norm heatmap 【高優先，10 分鐘】

取數張 **normal 影像**，畫出 patch token 的 L2-norm heatmap。

- ViT 會產生 high-norm artifact tokens（*Vision Transformers Need Registers*, ICLR 2024）
- 若 1M continued-pretrain 未正確處理 register tokens，這些異常 token 會在 feature map 上形成**固定位置的高響應熱點**
- 這些熱點**在 normal 影像上也一樣亮** → 是高信心 FP 的完美來源

**檢查點**：把 heatmap 與實際 FP 的位置疊圖比對。若對得上，這是 bug，修掉即可。

---

### 3.3 FP 樣本檢視 【半天】

1. 將 DINO 的 FP **按分數由高到低全部拉出目視**（量級約 150 張）
2. 用 DINOv3 CLS embedding 對這些 FP 做分群
3. 同時檢視 DINO 漏掉、YOLO 抓到的 FN（數十張）

預期會集中在少數幾個 cluster（charging artifact、特定 pattern、wafer edge、focus drift 等）。知道是哪幾個 cluster，後續修法可以非常有針對性。

---

### 3.4 SimpleFPN 的輸入層 【設定確認】

確認 SimpleFPN 吃的是 backbone 的**哪幾層**。

若僅使用最後一層（ViTDet 原始做法），可改為 concat 多層（ViT-B 常見取 layer 3 / 5 / 7 / 11）。淺層保留較多高頻與局部資訊，而高頻正是判別 defect / nuisance 所需。

---

## 4. 潛在解決方法

> 排序依據：預期效果 / 改動成本。前兩項同時對應根因 A 與 B。

### 4.1 【最高優先】輸入解析度 ×2

直接對應**根因 A**。

- DINOv3 支援 high-resolution adaptation，僅需**內插 position embedding**
- token map 密度提升 4 倍，缺陷從 <2 個 token 變成 ~6 個 token
- latency 不設限，此路徑成本可接受

**替代方案**：若記憶體不足，改用 **ViT-S/16 @ 2× 解析度**，通常優於 **ViT-B/16 @ 1× 解析度**。在小物件任務上，解析度的邊際效益幾乎總是大於模型容量。

---

### 4.2 【最高優先】以原生解析度 random crop 取代 mosaic

直接對應**根因 B**，且不付根因 A 的代價。

訓練時在**原生解析度**下隨機裁切 patch，推論時採用重疊 tiling 合併。

| | 打斷全域捷徑 | 解析度代價 | 是否 domain-specific 調參 |
|---|---|---|---|
| Mosaic | ✅ | ❌ 物件尺度砍半 | 否 |
| **Random crop（原生解析度）** | ✅ | ✅ 無 | 否 |

**這也是驗證根因 B 的乾淨實驗。** Mosaic 的失敗混合了兩個相反效應（傷解析度 vs 可能打斷捷徑），因此無法歸因；random crop 移除了解析度這個變數，讓「打斷全域捷徑」單獨可測。

- 若 image-level FP 下降 → **根因 B 成立**，此路線即為解法
- 若無變化 → 根因 B 可排除，資源全部押到 4.1

---

### 4.3 【零成本】先測 mosaic 版本的 image-level 指標

模型已存在，不需訓練、不需寫新 code，因此**仍值得跑**，但定位是**診斷**而非解法。

由於 mosaic 在 bbox test 上全面劣化（2.4），**不預期它在 image-level 會勝出**。它的價值在於提供根因 B 的第一個訊號：

| 觀察 | 推論 |
|---|---|
| image-level FP **明顯低於** non-mosaic 版本 | 根因 B（全域捷徑）真實存在，且效果足以蓋過解析度損失 → 4.2 值得優先投入 |
| image-level 也一併變差 | 解析度損失主導一切 → 根因 A 是唯一主線，資源押 4.1 |

無論哪個結果，都能在零成本下縮小後續搜尋空間。

---

### 4.4 凍結 backbone / LoRA ablation

一次訓練，資訊量最大：

| 結果 | 結論 | 下一步 |
|---|---|---|
| FP 明顯下降 | 50 張 fine-tune 造成特徵漂移 | 改用 LoRA 或大幅降低 backbone lr |
| FP 未下降 | 排除漂移假說 | 資源全部押到 4.1 / 4.2 |

**凍結 backbone 意味著更少的可調參數，是減少調參自由度而非增加。**

---

### 4.5 高頻 CNN 側支路（架構改動）

若 4.1 / 4.2 後仍有落差，考慮此項。

在 SimpleFPN 旁並接一條輕量 conv stem（3~4 層，stride 2/4，數百 K 參數），直接從原圖抽取 stride-4 特徵，與 ViT 上採樣的 p2 融合（concat + 1×1 conv，或 cross-attention）。

- ViT 負責語意與 context，CNN 負責高頻細節
- 參數量極小，50 張影像也訓得動
- 此為 Dino U-Net / SegDINO / DINO-AugSeg 等 2025 年工作的共同設計模式

> 參考：多篇 DINOv3 研究一致指出，天真地以固定 1×1 conv 傳遞 DINOv3 特徵會明顯損害 dense prediction 精度與邊界定位；但搭配適當的 feature adaptation module 後，即使凍結 backbone 也能達到 SOTA 表現。**SimpleFPN 基本上就是最天真的那種接法。**

---

### 4.6 `close_mosaic` 的單位修正（若保留 mosaic 路線）

`close_mosaic` 的單位是 epoch，而 epoch 大小與資料集規模成正比。沿用 COCO 的預設值「10」，在 50 張資料上等於把 clean 階段從 ~18,500 steps 縮到 ~60 steps。

若保留 mosaic，應將 `close_mosaic` 拉到總 epoch 的 **30~50%**，並**對 DINO 與 YOLOv26-50FS 兩邊同步修改**（此偏差對兩者相同，同步修改不影響比較公平性）。

補充：`scale` 由 0.5 降至 0.2 亦有物理依據 —— SEM 為固定倍率，缺陷具有真實物理尺寸先驗，1.5× 的尺度抖動在此 domain 不合理。

---

## 5. 執行順序

| # | 動作 | 類型 | 成本 |
|---|---|---|---|
| 1 | 測 mosaic 版本的 image-level 指標（4.3，**診斷用**） | 評估 | 零 |
| 2 | Normalization 一致性檢查（3.1） | 除錯 | 10 min |
| 3 | Token norm heatmap 檢查（3.2） | 除錯 | 10 min |
| 4 | FP 樣本目視 + 分群（3.3） | 診斷 | 半天 |
| 5 | **輸入解析度 ×2（4.1）** | 訓練 | 1 run |
| 6 | **Random crop 取代 mosaic（4.2）** | 訓練 | 1 run |
| 7 | 凍結 backbone ablation（4.4） | 訓練 | 1 run |
| 8 | SimpleFPN 多層特徵（3.4） | 訓練 | 1 run |
| 9 | 高頻 CNN 側支路（4.5） | 架構 | 中 |

**#1~#4 不需訓練，可在一天內完成。**

**#5（解析度 ×2）是主線** —— 根因 A 是目前唯一有直接證據支持的根因，mosaic 實驗的全面劣化更進一步佐證了它。

**#6（random crop）是根因 B 的唯一乾淨驗證** —— 若 #1 的結果指向根因 B 存在，則將 #6 提前到與 #5 同等優先。

---

## 6. 限制條件（agent 必須遵守）

- ❌ **不新增任何標註資料**
- ❌ **不做 per-dataset 的超參數搜尋** —— 所有改動必須有物理或架構層面的理由，不得靠 test set 反覆試誤
- ❌ **不在 test set 上調整任何參數**
- ✅ **維持架構公平性**：DINO 與 YOLOv26 使用相同 decoder、相同標註量（50 張）、相同評估協定
- ✅ **Recipe 與 backbone 側的架構選擇不視為不公平** —— 每個 backbone 使用其適配的訓練配方是文獻慣例（layer decay 已是此類選擇）
- ✅ 任何 augmentation / schedule 的修改，**若同時適用於 YOLOv26-50FS，則兩邊同步套用**
- ⚠️ latency 暫不列入考量（先求好再求快）

---

## 7. 參考文獻

| 主題 | 文獻 |
|---|---|
| DINOv3 | Siméoni et al., *DINOv3*, arXiv:2508.10104, 2025 |
| ViT artifact tokens | Darcet et al., *Vision Transformers Need Registers*, ICLR 2024 |
| ViT detection baseline | Li et al., *Exploring Plain Vision Transformer Backbones for Object Detection* (ViTDet), ECCV 2022 |
| DINOv3 dense adapter 設計 | Dino U-Net (2025) / SegDINO (2025) / DINO-AugSeg (2026) |
| DINOv3 few-shot detection | *DINOv3 for DE-ViT: Boosting Few-Shot Object Detection*, 2026 |
| 工業異常偵測（若日後解禁 negative 使用） | Dinomaly (CVPR 2025) / INP-Former (CVPR 2025) / AnomalyDINO (WACV 2025) |
| SEM ADC + DINOv2 | Zhang et al., *Semiconductor SEM Image Defect Classification with ViT*, ASMC 2025 (arXiv:2506.03345) |

---

## 附錄：目前被排除但值得日後解禁的方向

以下方向預期效果顯著，但因「不做 data engineering」與「維持架構公平性」的限制而暫緩。若 §5 執行後仍未達標，建議依序解禁：

1. **從 1M SSL pool 挖 negative images** —— defect rate 2~6% 表示該 pool 本身即為純度 94~98% 的 normal 集合，可直接作為 background image（所有 query 監督為 ∅）加入訓練。**公平做法：對 DINO 與 YOLOv26-50FS 同時加入同一批 negative 重訓。** 這是目前最大的未使用資源。
2. **Hard negative mining** —— 訓練後回掃 1M，取分數最高的 normal 加權重訓，迭代 2~3 輪。
3. **Copy-paste augmentation** —— 將 50 個 GT 缺陷貼至 normal 背景，同時擴增正樣本與背景多樣性。
4. **5-fold ensemble + TTA** —— 50 train + 50 val 共 100 張，以不同 80/20 切分訓 5 個模型平均。高信心 FP 通常是 model-specific 的，ensemble 會將其平均掉，而真缺陷是所有模型都同意的 —— 機制上直接對應「保 TP、砍 FP」。
5. **二階段 verifier** —— DINO 開低門檻出框 → crop（含 2~3× context）→ 凍結 DINOv3 → linear probe / kNN；負樣本取自挖掘出的 FP crops。
