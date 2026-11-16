# G-swap & offset-concept experiments trên Infinity-2B

Ngày chạy: 2026-11-16 · Nhánh: `module_space_prompt_offset` · GPU: A100-40GB

## 0. Câu hỏi

Infinity đưa text vào mô hình qua ba route:

```
pooled text ─► cond_BD ─┬─► SOS token
                        └─► G = shared_ada_lin(cond_BD)   (6×C AdaLN: gamma1/2, scale1/2, shift1/2; dùng chung mọi block)
token text  ─► text_proj_for_ca ─► cross-attention K/V
```

(head AdaLN `head_nm` đọc `cond_BD`, không đọc G.)

Paper (Fig. 11) cần biết **G có mang ngữ nghĩa không**: nếu chỉ đổi G, với SOS + cross-attention cố định, thì thứ được sinh ra có đổi không? Sau đó: **A/A+ (offset trong không gian G) có phải là nơi "tinh chỉnh identity dựa trên một concept tổng quát đã có" không?**

## 1. Cách cài đặt

- **Can thiệp G không sửa repo:** dùng `forward_pre_hook(with_kwargs=True)` trên mọi block, thay `cond_BD` theo `scale_ind` (`g_swap.py`).
- **CFG (cfg=4, tau=0.5, 1024², top_k=900, top_p=0.97):** chỉ nửa *cond* của batch bị thay G; nửa *uncond* giữ G của null text.
- **G_c** được tính từ prompt `"a photo of a {c}"`.
- **flash_attn:** không build được cho torch 2.14/cu132, nên dùng shim pure-PyTorch (`shim/flash_attn`). Self-attention vốn đã dùng SDPA (customized flash tắt); chỉ cross-attention varlen cần shim.
- **Scale schedule 1M (13 scale):** 1,2,4,6,8,12,16,20,24,32,40,48,64. "Early S" = k scale đầu.

## 2. E0 — Fig. 11: chỉ đổi G, prompt cố định "a photo"

4 seed; G_cat / G_dog áp ở mọi scale; nội suy G(t) = (1−t)G_cat + t·G_dog với t ∈ {0, .25, .5, .75, 1} trên k ∈ {4, 5, 6, 13} scale đầu.

**Kết quả: G một mình không đổi class.**
- Mọi điều kiện đều ra cùng chủ thể như prompt neutral (một người phụ nữ). Layout giữ nguyên, chỉ chi tiết đổi (kính râm, túi, quần áo).
- Ngoại lệ duy nhất: seed 3 + G_cat ra một sinh vật lông giống *chó*, cùng pose và layout.
- G(t) không cho hỗn hợp cat/dog mượt; t chỉ làm nhiễu chi tiết một cách hỗn loạn.
- k = 4, 5, 6, 13 cho ảnh gần như giống hệt nhau (early4 t=1 ≡ G_dog mọi scale). Vậy ảnh hưởng nhỏ của G được "khóa" trong 4 scale đầu (≤ 6×6).
- **Nguyên nhân:** cos(G_cat, G_dog) = 0.9999993; ‖G_cat − G_dog‖ / ‖G_n‖ = **0.12%**. Bias của `shared_ada_lin` áp đảo, pooled text gần như không làm G dịch chuyển.

**Control:**

| Điều kiện | Kết quả |
|---|---|
| Prompt "a photo of a cat/dog" cho SOS+CA, **G = G_neutral** | Gần như trùng ảnh full prompt → class đi hoàn toàn qua SOS + CA |
| G_n + α(G_c − G_n), α = 5 | Vẫn là người |
| α = 20 | Ảnh vỡ thành texture trừu tượng |

Không có độ khuếch đại nào biến hướng class trong G thành class.

**Đoạn đề xuất cho placeholder trong paper:**
> Changing G alone does not change the class: with SOS and cross-attention fixed to "a photo", G_cat and G_dog yield the same subject and layout, differing only in small details, and G(t) produces no gradual mixture. The reverse swap (class prompt in SOS and cross-attention, G fixed to the neutral prompt) reproduces the class almost exactly. Text-derived G differs only marginally between classes (cos 0.9999993). Identity in the pretrained model is therefore carried by the SOS and cross-attention routes, and the effect of A+ comes from the optimized offset reaching a region of G-space that text prompts do not, rather than from G as computed from text.

Câu kết "G encodes what is generated, while cross-attention controls how it is placed" **không được kết quả này ủng hộ** và cần viết lại.

## 3. E1 — G từ text có mang thuộc tính không?

SOS + CA = "a photo of a cat". G lấy từ prompt thuộc tính, G = G_base + α(G_attr − G_base), áp mọi scale, 2 seed. Điểm: CLIP ViT-L/14, P(attr) = softmax giữa "a photo of a cat" và prompt thuộc tính, lấy trung bình 2 seed.

| Thuộc tính | ‖ΔG‖/‖G‖ | base | full prompt | qua SOS+CA (G=base) | chỉ G α=1 | α=5 | α=10 |
|---|---|---|---|---|---|---|---|
| black cat | 0.25% | 0.00 | 1.00 | 1.00 | 0.01 | 0.00 | 0.00 |
| young kitten | 0.29% | 0.28 | 0.90 | 0.90 | 0.27 | 0.25 | 0.20 |
| fluffy Persian cat | 0.44% | 0.04 | 0.99 | 0.99 | 0.08 | 0.14 | 0.19 |
| watercolor painting | 0.55% | 0.00 | 0.42 | 0.12* | 0.00 | 0.00 | 0.00 |

\* Nhìn ảnh vẫn là tranh màu nước, gần trùng full prompt; CLIP 2-way chấm thấp hơn.

**Kết luận:**
- Thuộc tính, giống class, đi **hoàn toàn qua SOS + CA**.
- G từ text gần như là một điểm cố định (cos ≥ 0.99998). Hiệu ứng duy nhất thấy được là Persian ở α=10 lông hơi xù hơn.
- Vì vậy không gian G mà text tạo ra **không dùng được** để giải thích A/A+. Offset A/A+ đã học là một hướng mà text không bao giờ tạo ra.

## 4. E3 — Offset A+ đã học có cần concept nền không?

- **Train:** offset per-scale trên scale 0–5 (giống A+), 5 ảnh DreamBooth `dog` (một chú corgi), train prompt "a dog", 60 bước, 512px (0.25M, 10 scale), lr 5e-3, bs 2. bit_acc tăng từ 67% lên 79%. ‖offset‖ mỗi scale ≈ 4.9–6.3.
- **Infer:** cùng offset với 5 base prompt; offset ×0 và ×1; cfg=1; 3 seed.

| Base prompt | offset ×0 | offset ×1 | Diễn giải |
|---|---|---|---|
| "a dog" | chó bất kỳ | **đúng con corgi** | concept khớp → ra identity |
| "a photo" | người / cảnh phố | **vỡ**: khối lông, texture, không ra con corgi | không có concept → offset không tự tạo subject |
| "a photo of a cat" | mèo | **vẫn là mèo**, mang màu cam-trắng và lông xù của corgi | class giữ nguyên, thuộc tính được chuyển sang |
| "a dog on the beach" | chó ở bãi biển | corgi, **mất bãi biển**, nền giống ảnh train | identity đúng, ngữ cảnh bị ghi đè |
| "a dog in a bucket" | chó trong xô | corgi, cái xô gần như mất | như trên |

**Kết luận (sơ bộ):**
1. **Offset cần concept nền.** Không có "dog" trong prompt thì không ra subject. Điều này khớp với E0/E1: class đi qua SOS + CA.
2. **Offset hoạt động như một hướng instance/thuộc tính bên trong concept.** Áp lên "cat" thì ra mèo mang ngoại hình corgi, đúng kiểu "cat → black/young cat" nhưng ở mức instance.
3. **Offset ở scale 0–5 còn mang theo layout và nền của ảnh train**, nên ghi đè ngữ cảnh của prompt (beach, bucket). A+ ở scale sớm chưa thuần identity.

**Giả thuyết tổng hợp:** SOS + cross-attention quyết định *cái gì* (class/concept, thuộc tính diễn đạt được bằng lời) và *ở đâu*. G gần như hằng với text. Offset A/A+ là một hướng **ngoài vùng G-từ-text**, tinh chỉnh *instance nào* trong concept đã có, nhưng ở scale sớm thì kéo theo cả bố cục.

## 5. Hạn chế

- **E3** là bản chạy nhanh: 1 subject, 60 bước, 512px, cfg=1, 3 seed, chưa có metric định lượng. Offset ×1 có artifact (viền khung, ảnh hơi vỡ), có thể do strength quá mạnh hoặc train chưa đủ lâu.
- **E0/E1:** chỉ 2–4 seed, một cặp class (cat/dog), một bộ 4 thuộc tính.
- **Định nghĩa G_c:** G_c lấy từ "a photo of a {c}". Chưa thử G từ tên class trần ("cat").
- **Shim flash_attn:** chậm hơn bản gốc. Về toán học là tương đương, nhưng chưa so số học với flash_attn thật.

## 6. Bước tiếp theo đề xuất

1. **E3 sweep:** offset strength {0.25, 0.5, 0.75} × scale {0–3, 0–5}. Mục tiêu là tìm vùng giữ được identity mà vẫn giữ "beach"/"bucket", tức tách identity khỏi layout. Đo DINO/CLIP-I so với ảnh ref, CLIP-T với prompt, SSIM layout so với ảnh offset ×0.
2. **E3 nhiều subject** (≥ 5 lớp DreamBooth), cfg 1 và 4, train dài hơn (≥ 200 bước, 1M).
3. **Train offset với null text** (`TRAIN_PROMPT=""`), rồi lặp lại bảng E3. Câu hỏi: offset học không cần concept thì có tự mang concept không?
4. **E2:** chuyển ΔG thuộc tính hoặc offset giữa các class (corgi-offset → "a fox", "a teddy bear") để đo mức độ class-agnostic.
5. **E5:** so ‖offset‖ với ‖ΔG_text‖ (≈ 0.1–0.5% ‖G‖) và chiếu offset lên PCA của G-từ-text, để xác nhận định lượng rằng offset nằm ngoài vùng text.

## 7. Tái tạo

```bash
cd Modulation-Edit
# deps (ngoài torch): easydict typed-argument-parser timm==0.9.6 transformers sentencepiece opencv-python-headless
#                    einops omegaconf kornia pandas pyyaml pytz seaborn gputil colorama imageio wandb decord
# weights/: infinity_2b_reg.pth, infinity_vae_d32reg.pth (FoundationVision/infinity), flan-t5-xl (google/flan-t5-xl)

bash scripts/run_g_swap_class.sh      # E0 / Fig. 11  (~17 phút, 125 ảnh 1024²)  -> output/g_swap/e0
bash scripts/run_g_swap_attr.sh       # E1            (~7 phút)                  -> output/g_swap/e1
bash scripts/run_offset_concept.sh    # E3            (~4 phút)                  -> output/g_swap/e3
```

E3 cần `data/dreambooth/dataset/dog/` (5 ảnh `00.jpg`–`04.jpg` từ `github.com/google/dreambooth/tree/main/dataset/dog`).

| File | Vai trò |
|---|---|
| `experiments/g_swap/g_swap.py` | E0: hook thay G per-scale, nội suy, control |
| `experiments/g_swap/make_fig.py` | Ghép Fig. 11, sweep \|S\|, hình control |
| `experiments/g_swap/e1_attr.py` / `e1_eval.py` | E1: sinh ảnh + chấm CLIP + lưới ảnh |
| `experiments/g_swap/shim/flash_attn/` | Thay thế flash_attn bằng PyTorch thuần |
| `scripts/run_g_swap_class.sh`, `run_g_swap_attr.sh`, `run_offset_concept.sh` | Lệnh chạy |
| `output/g_swap/{e0,e1,e3}/` | Kết quả (gitignored): `fig11_early{4,5,6,13}.jpg`, `fig_controls.jpg`, `e1_grid_seed*.jpg`, `clip_scores.json`, `e3_infer.jpg`, `offset.pth` |
