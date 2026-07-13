# Skill: model-selection

# Выбор моделей для ik_llama.cpp (9950X + RTX 3090)

## Когда использовать

Активируй этот skill когда нужно:
- Подобрать GGUF модель для сборки ik_llama.cpp на 9950X + RTX 3090
- Определить оптимальный квант для модели
- Переквантовать модель в SOTA кванты (IQ*_K, IQ*_KS, IQ*_KT)
- Найти готовые GGUF модели на HuggingFace
- Оценить VRAM/RAM требования модели

## ГЛАВНЫЙ ПРИНЦИП: АППАРАТНОЕ УСКОРЕНИЕ = КРИТЕРИЙ №1

**АППАРАТНО** — это главное слово. Если квант не аппаратно ускорен — он ПЛОХОЙ.

### Два типа ускорения, оба обязательны:

```
1. ВЕСА МОДЕЛИ: IQ*_K / Q*_K кванты
   CPU: AVX-512 VNNI → _mm512_dpbusd_epi32 (int8×int8→int32)
   GPU: INT8 тензорные ядра → MMQ kernels

2. KV CACHE: -ctk / -ctv типы
   CPU: AVX-512 FA kernels (q8_0, q8_KV, q6_0, q4_0, q4_1, iq4_nl)
   GPU: dp4a CUDA cores (q8_0, q6_0, q4_0, q4_1, iq4_nl)
```

### Чеклист: "Этот квант аппаратно ускорен?"

```
Вопрос 1: Есть ли CPU ядро с AVX-512 VNNI?
  → IQ*_K серия: ДА (iqX_k_q8_K_AVX512 / _new)
  → Q*_K серия:  ДА (qX_K_q8_K_AVX512)
  → Q*_0 серия:  ЧАСТИЧНО (базовый AVX-512, без VNNI)
  → Q*_1 серия:  НЕТ (legacy)

Вопрос 2: Есть ли GPU ядро с MMQ?
  → IQ*_K серия: ДА (MMQ + MMVQ)
  → Q*_K серия:  ДА (MMQ + MMVQ)
  → Q*_0 серия:  ДА (MMQ + MMVQ)
  → Q*_1 серия:  НЕТ (только fattn-vec)

Вопрос 3: Есть ли FA kernel для KV cache?
  → q8_KV:       ТОЛЬКО CPU (нет GPU ядра!)
  → q8_0:        ДА (CPU + GPU) ← ЛУЧШИЙ ВЫБОР
  → q6_0:        ДА (CPU + GPU)
  → q4_0:        ДА (CPU + GPU, с ALL_QUANTS)
  → q4_1:        ДА (CPU + GPU, с ALL_QUANTS)
  → iq4_nl:      ДА (CPU + GPU, с ALL_QUANTS)
  → f16:         ДА (CPU + GPU, но без квантования)
  → q5_0/q5_1:   НЕТ (нет FA ядер)
```

### Золотое правило

```
ДЛЯ МОДЕЛИ: IQ4_KS / IQ4_K (аппаратно ускорены и на CPU, и на GPU)
ДЛЯ КЕША:   q8_0 (аппаратно ускорен и на CPU, и на GPU)

q8_KV — ТОЛЬКО для CPU (если модель чисто CPU)
q4_0 — агрессивный кеш (только если VRAM критичен)
```

### Комбинации веса + кеш: что аппаратно ускорено

| Веса модели | KV cache | CPU VNNI | GPU MMQ | GPU dp4a | Итого |
|-------------|----------|:--------:|:-------:|:--------:|:-----:|
| IQ4_KS | q8_0 | ✅ | ✅ | ✅ | **✅✅✅ ЛУЧШИЙ** |
| IQ4_K | q8_0 | ✅ | ✅ | ✅ | **✅✅✅ ОТЛИЧНЫЙ** |
| Q4_K | q8_0 | ✅ | ✅ | ✅ | ✅✅ ХОРОШИЙ |
| IQ4_KS | q6_0 | ✅ | ✅ | ✅ | ✅✅ ХОРОШИЙ |
| IQ4_KS | q4_0 | ✅ | ✅ | ✅ | ✅ ХОРОШИЙ |
| Q8_0 | q8_0 | ✅ | ✅ | ✅ | ✅ ПОЛНЫЙ 8-BIT |
| IQ4_KS | q8_KV | ✅ | ✅ | ❌ | ⚠️ CPU-only кеш |
| IQ4_KS | f16 | ✅ | ✅ | ✅ | ⚠️ Нет квантования кеша |
| Q4_0 | q4_0 | ✅ | ✅ | ✅ | ⚠️ Legacy веса |

**Читать таблицу:**
- `CPU VNNI` = веса ускорены через VNNI на CPU
- `GPU MMQ` = веса ускорены через MMQ на GPU
- `GPU dp4a` = KV cache ускорен через dp4a на GPU
- **Все три ✅ = максимальная скорость на 9950X + RTX 3090**

## Аппаратные ограничения

### CPU: AMD Ryzen 9 9950X (Zen 5)

| Инструкция | Поддержка | Влияние на кванты |
|------------|-----------|-------------------|
| AVX-512F | ✅ | Базовые SIMD операции |
| AVX-512VNNI | ✅ | Умножение int8×int8 → int32 (DP4A) — **ключевое для квантов** |
| AVX-512BW | ✅ | Байтовые операции |
| AVX-512DQ | ✅ | Целочисленные операции |
| AVX-512VL | ✅ | 128/256-bit versions |
| AVX-512BF16 | ✅ | Нативный BF16 dot product |
| FMA | ✅ | Fused multiply-add |

**Ключевой инструкция**: `_mm512_dpbusd_epi32` (VNNI) — умножает 16 пар unsigned int8 и накапливает в int32 за один цикл. Это делает Q8_*/IQ*_K кванты **крайне быстрыми** на CPU.

### GPU: NVIDIA RTX 3090 (sm_86, 24GB VRAM)

| Возможность | Значение |
|-------------|----------|
| Compute capability | 8.6 |
| CUDA cores | 10496 |
| Tensor cores | 328 (INT8, FP16, BF16) |
| VRAM | 24 GB GDDR6X |
| Пропускная способность | 936 GB/s |

**Ключевое**: INT8 тензорные ядра → MMQ kernels работают на **максимальной скорости** для всех квантов ≤8 бит.

## SOTA кванты: полный каталог

### ik_llama.cpp原创ные кванты (не существуют в mainline llama.cpp)

#### Нелинейные кванты (IQ*_K серия) — ЛУЧШЕЕ качество/бит

| Квант | bpw | Описание | CPU ядро | GPU ядро |
|-------|-----|----------|----------|----------|
| **IQ2_KS** | 2.1875 | 2-bit, super-block 32, newest kernel | `iqX_k_q8_K_AVX512_new` | MMQ + MMVQ |
| **IQ2_K** | 2.375 | 2-bit, non-linear mapping | `iqX_k_q8_K_AVX512` | MMQ + MMVQ |
| **IQ2_KL** | 2.69 | 2-bit, large variant | `iqX_k_q8_K_AVX512_new` | MMQ + MMVQ |
| **IQ3_KS** | 3.19 | 3-bit, super-block 32 | `iqX_k_q8_K_AVX512_new` | MMQ + MMVQ |
| **IQ3_K** | 3.44 | 3-bit, non-linear mapping | `iqX_k_q8_K_AVX512` | MMQ + MMVQ |
| **IQ4_KSS** | 4.0 | 4-bit, supersmall variant | `iqX_k_q8_K_AVX512_new` | MMQ + MMVQ |
| **IQ4_KS** | 4.25 | 4-bit, super-block 32 | `iqX_k_q8_K_AVX512_new` | MMQ + MMVQ |
| **IQ4_K** | 4.5 | 4-bit, non-linear mapping | `iqX_k_q8_K_AVX512` | MMQ + MMVQ |
| **IQ5_KS** | 5.25 | 5-bit, super-block 32 | `iqX_k_q8_K_AVX512_new` | MMQ + MMVQ |
| **IQ5_K** | 5.5 | 5-bit, non-linear mapping | `iqX_k_q8_K_AVX512` | MMQ + MMVQ |
| **IQ6_K** | 6.6 | 6-bit, non-linear mapping | `iqX_k_q8_K_AVX512` | MMQ + MMVQ |

#### R4/R8 repacked варианты (тот же качество, +15-30% CPU PP)

| Квант | bpw | Примечание |
|-------|-----|------------|
| IQ2_K_R4 | 2.375 | R4 repack для IQ2_K |
| IQ3_K_R4 | 3.44 | R4 repack для IQ3_K |
| IQ4_K_R4 | 4.5 | R4 repack для IQ4_K |
| IQ4_KS_R4 | 4.25 | R4 repack для IQ4_KS |
| IQ5_K_R4 | 5.5 | R4 repack для IQ5_K |
| IQ5_KS_R4 | 5.25 | R4 repack для IQ5_KS |

#### Trellis кванты (IQ*_KT серия) — +0.2 bpw точность

| Квант | bpw | Описание |
|-------|-----|----------|
| IQ1_KT | 1.75 | 1-bit trellis |
| IQ2_KT | 2.125 | 2-bit trellis |
| IQ3_KT | 3.125 | 3-bit trellis |
| IQ4_KT | 4.0 | 4-bit trellis |

#### 1-bit / Bitnet кванты

| Квант | bpw | Описание |
|-------|-----|----------|
| IQ1_BN | 1.62 | Bitnet ternary (-1, 0, +1) |
| IQ2_BN | 2.0 | Bitnet 2-bit |
| IQ1_S | 1.56 | 1-bit, requires imatrix |
| IQ1_M | 1.75 | 1-bit, better than IQ1_S |
| Q1_0_G128 | 1.0 | 1-bit, group 128 |

### Стандартные K-quant кванты (upstream llama.cpp)

| Квант | bpw | CPU ядро | GPU ядро |
|-------|-----|----------|----------|
| Q2_K | ~2.5 | `qX_K_q8_K_AVX512` | MMQ + MMVQ |
| Q3_K | ~3.0 | `qX_K_q8_K_AVX512` | MMQ + MMVQ |
| Q4_K_S/M | ~3.6/3.8 | `qX_K_q8_K_AVX512` | MMQ + MMVQ |
| Q5_K_S/M | ~4.4/4.5 | `qX_K_q8_K_AVX512` | MMQ + MMVQ |
| Q6_K | ~5.15 | `qX_K_q8_K_AVX512` | MMQ + MMVQ |
| Q8_0 | ~8.5 | `q8_0` (простой) | MMQ + MMVQ |
| Q6_0 | 6.5 | Legacy kernel | fattn-vec |

### Legacy кванты

| Квант | bpw | Примечание |
|-------|-----|------------|
| Q4_0 | 3.56 | Самый быстрый legacy |
| Q4_1 | 3.91 | С offset |
| Q5_0 | 4.33 | С 1 extra bit |
| Q5_1 | 4.70 | С offset + extra bit |
| IQ4_NL | 4.5 | Non-linear legacy |
| IQ4_XS | 4.25 | Non-linear, super-block |

## Сравнение качества (обсуждение #8)

| bpw | Лучший квант | Ошибка vs Q8_0 | Улучшение vs standard |
|-----|-------------|----------------|----------------------|
| 2.0 | IQ2_KS | ~3.5% | IQ2_KS > IQ2_K > IQ2_S > Q2_K |
| 3.0 | IQ3_KS | ~2.0% | IQ3_KS > IQ3_K > IQ3_S > Q3_K |
| 4.0 | IQ4_KSS | ~1.2% | IQ4_KSS > IQ4_KS > IQ4_K > Q4_K |
| 4.5 | IQ4_K | ~0.9% | IQ4_K > Q4_K (в 2.7× меньше ошибки, чем Q4_0) |
| 5.25 | IQ5_KS | ~0.6% | IQ5_KS > IQ5_K > Q5_K |
| 5.5 | IQ5_K | ~0.5% | IQ5_K > Q5_K (в 2.1× меньше ошибки, чем Q5_0) |
| 6.6 | IQ6_K | ~0.15% | IQ6_K > Q6_K (в 4× меньше ошибки) |

## Рекомендации по подбору

### Решение: какой квант выбрать

```
Есть importance matrix?
├── ДА → Используй IQ*_KS/IQ*_K с --imatrix
│   ├── Нужно ≤2.5 bpw → IQ2_KS + imatrix
│   ├── Нужно ≤3.5 bpw → IQ3_KS + imatrix
│   ├── Нужно ≤4.5 bpw → IQ4_KS + imatrix  ← РЕКОМЕНДУЕМЫЙ
│   ├── Нужно ≤5.5 bpw → IQ5_KS + imatrix
│   └── Нужно ≤7.0 bpw → IQ6_K
│
└── НЕТ → Используй Q*_K_S/M или IQ*_K без imatrix
    ├── Нужно ≤3.5 bpw → Q3_K_M
    ├── Нужно ≤4.5 bpw → Q4_K_M  ← БЕЗОПАСНЫЙ ВЫБОР
    ├── Нужно ≤5.5 bpw → Q5_K_M
    └── Нужно ≤7.0 bpw → Q6_K
```

### Оценка VRAM для RTX 3090 (24GB)

```
Размер модели × bpw / 8 = VRAM для весов

Примеры:
  7B × 4.25 / 8 = 3.7 GB  → IQ4_KS на 7B: ✅ влезет + KV cache
  13B × 4.25 / 8 = 6.9 GB → IQ4_KS на 13B: ✅ влезет + KV cache
  34B × 4.25 / 8 = 18 GB  → IQ4_KS на 34B: ✅ влезет, тесно
  70B × 4.25 / 8 = 37 GB  → IQ4_KS на 70B: ❌ не влезет, нужен CPU offload

Для 70B на RTX 3090:
  → Используй --fit flag (автоматический split)
  → Или IQ4_KSS (4.0 bpw) = 35 GB — всё ещё не влезет
  → Или Q3_K_M (3.0 bpw) = 26 GB — почти влезет
  → Или IQ2_KS (2.2 bpw) = 19 GB — влезет, но качество низкое
```

### Рекомендации для 9950X + RTX 3090

| Сценарий | Модель | Квант | Ожидаемая производительность |
|----------|--------|-------|----------------------------|
| **Чат-бот 7B** (макс. скорость) | Qwen2.5-7B / Llama-3.1-8B | IQ4_KS | PP 2000+, TG 80+ t/s |
| **Чат-бот 13B** (баланс) | Qwen2.5-14B / Llama-3.1-13B | IQ4_KS | PP 1200+, TG 50+ t/s |
| **Кодинг 7B** | DeepSeek-Coder-V2-Lite | IQ4_KS | PP 1800+, TG 70+ t/s |
| **Чат-бот 34B** (качество) | Qwen2.5-32B | IQ4_K | PP 600+, TG 25+ t/s |
| **Чат-бот 70B** (макс. качество) | Llama-3.1-70B | Q3_K_M (--fit) | PP 300+, TG 15+ t/s |
| **MoE 14B×3B** | Qwen2-5-MoE-A2.7B | IQ4_KS | PP 2500+, TG 100+ t/s |
| **MoE 57B×14B** | DeepSeek-V2-Lite | IQ4_K | PP 800+, TG 30+ t/s |

## Где искать GGUF модели

### HuggingFace репозитории

| Автор | Что искать | Примечание |
|-------|-----------|------------|
| **bartowski** | `bartowski/<model>-GGUF` | Лучший выбор, все кванты, imatrix |
| **mradermacher** | `mradermacher/<model>-GGUF` | Хороший выбор, все кванты |
| **unsloth** | `unsloth/<model>-GGUF` | ⚠️ Избегай моделей с `_XL` в имени |
| **ik_llama.cpp** | `ikawrakow/<model>-GGUF` | Официальные кванты для ik_llama |

### Поиск на HuggingFace

```bash
# Через CLI
pip install huggingface_hub
huggingface-cli search "Qwen2.5-32B-GGUF IQ4_KS"

# Через API
curl "https://huggingface.co/api/models?search=Qwen2.5-32B+GGUF&sort=downloads"
```

### Критерии выбора модели

1. **Импорт**: Проверь что модель указана в `ik_llama.cpp/README.md` (раздел "Supported models")
2. **Квант**: Ищи `IQ4_KS` или `IQ4_K` (SOTA качество)
3. **Imatrix**: Для квантов <4 bpw **обязательно** нужна importance matrix
4. **Размер**: Проверь что VRAM/RAM хватает
5. **Архитектура**: MoE модели эффективнее dense при том же качестве

## Переквантование в SOTA

### Использование llama-quantize

```bash
# Базовое квантование (без imatrix)
llama-quantize model-f16.gguf model-iq4_ks.gguf IQ4_KS

# С importance matrix (рекомендуется для <4 bpw)
llama-quantize --imatrix imatrix.dat model-f16.gguf model-iq3_ks.gguf IQ3_KS

# Кастомное квантование по тензорам
llama-quantize --custom-q "blk\.0\.\*=Q8_0" model-f16.gguf model-mixed.gguf IQ4_KS

# Repack в R4 вариант (ускорение CPU PP на 15-30%)
llama-quantize --repack model.gguf model-r4.gguf q8_0_r8

# Частичное переквантование (только изменённые тензоры)
llama-quantize --partial-requant old.gguf new.gguf IQ4_K
```

### Получение importance matrix

```bash
# Используй calibration dataset (typically 100-1000 примеров)
# importance matrix нужна для точного квантования < 4 bpw

# Пример: генерация imatrix через llama-imatrix
llama-imatrix -m model-f16.gguf -f calibration.txt -o imatrix.dat

# Или скачай готовую с HuggingFace
# Ищи файлы *.imatrix.dat в репозиториях bartowski/mradermacher
```

### Кастомное квантование по тензорам

Для максимального качества можно задать разные кванты для разных частей модели:

```bash
llama-quantize \
    --custom-q "output=Q8_0" \
    --custom-q "token_embd=Q6_K" \
    --custom-q "blk\.0\.attn_k=q6_K" \
    --custom-q "blk\.0\.attn_v=q6_K" \
    --custom-q "blk\.\*\.ffn_down=IQ4_KS" \
    model-f16.gguf model-optimized.gguf IQ4_KS
```

Правила:
- `output` тензор → всегда Q6_K или Q8_0 (критичен для качества)
- `token_embd` → Q6_K или Q8_0 (критичен для входных данных)
- `attn_k`, `attn_v` → Q6_K (важны для attention)
- `ffn_down` → IQ4_KS (наименее критичен, экономия места)

## Производительность на 9950X + RTX 3090

### CPU-only (AVX-512 VNNI)

| Квант | Prompt Processing | Token Generation |
|-------|-------------------|------------------|
| Q8_K_R16 | 100% (baseline) | 100% (baseline) |
| IQ4_KS_R8 | ~95% | ~95% |
| Q4_K | ~85% | ~90% |
| IQ4_K | ~90% | ~92% |
| IQ3_KS | ~75% | ~80% |
| IQ2_KS | ~50% | ~55% |

### GPU-only (RTX 3090 MMQ)

| Квант | Prompt Processing | Token Generation |
|-------|-------------------|------------------|
| Q8_0 | 100% (baseline) | 100% (baseline) |
| IQ4_KS | ~96% | ~97% |
| IQ4_K | ~94% | ~95% |
| Q4_K | ~90% | ~92% |
| IQ3_KS | ~85% | ~88% |
| IQ2_KS | ~70% | ~75% |

**Важно**: Token generation **всегда memory-bandwidth bound**. Поэтому:
- Низкий bpw = быстрее TG (меньше данных читать)
- IQ4_KS (4.25 bpw) — лучший баланс качество/скорость
- IQ2_KS (2.2 bpw) — максимальная скорость TG, но низкое качество

## Типичные ошибки

### 1. Использование квантов <4 bpw без imatrix

```
❌ llama-quantize model-f16.gguf model.gguf IQ2_KS
✅ llama-quantize --imatrix imatrix.dat model-f16.gguf model.gguf IQ2_KS
```

### 2. Использование Q4_0 вместо IQ4_KS

```
❌ Q4_0 (3.56 bpw, legacy, без нелинейного маппинга)
✅ IQ4_KS (4.25 bpw, SOTA, нелинейный маппинг, VNNI-optimized)
```

Q4_0 проигрывает IQ4_KS в качестве при **почти одинаковом** размере.

### 3. Использование q8_KV для KV cache на GPU

```
❌ -ctk q8_KV -ctv q8_KV  (нет GPU ядра! Только для CPU-only моделей)
✅ -ctk q8_0 -ctv q8_0    (аппаратно ускорен и на CPU, и на GPU)
```

q8_KV — самый быстрый на CPU, но **не работает на GPU**. Если модель в GPU (-ngl > 0), используй q8_0.

### 4. Использование q5_0/q5_1 для KV cache

```
❌ -ctk q5_0 -ctv q5_0  (нет FA ядер! Не работает с --flash-attn)
✅ -ctk q6_0 -ctv q6_0  (есть FA ядра, аппаратно ускорен)
```

q5_0/q5_1 **не имеют** flash attention kernel support.

### 5. Игнорирование output тензора

```
❌ --custom-q "ffn_down=IQ4_KS"  (output тензор остался Q4_0)
✅ --custom-q "output=Q6_K,ffn_down=IQ4_KS"
```

Output тензор **всегда** должен быть ≥Q6_K.

### 6. Загрузка 70B модели без --fit

```
❌ llama-server -m 70B.gguf  (VRAM overflow, crash)
✅ llama-server --fit -m 70B.gguf  (автоматический CPU/GPU split)
```

Для моделей >24GB на RTX 3090 **всегда** используй `--fit`.

## MOE (Mixture of Experts): полное руководство

### Поддерживаемые MOE архитектуры

| Архитектура | Модели | Graph Parallel |
|-------------|--------|----------------|
| DeepSeek-V3/V2 | DeepSeek-R1, DeepSeek-V3 | ✅ |
| Qwen3 MoE | Qwen3-30B-A3B, Qwen3-235B-A22B | ✅ |
| Qwen3.5 MoE | Qwen3.5-Turbo | ✅ |
| GLM-4 MoE | GLM-4-9B-Chat-1M | ✅ |
| Mistral 3/4 | Mistral-Small-24B | ✅ |
| Command-R / Cohere2 | Command-R-35B | ✅ |
| Hunyuan MoE | Hunyuan-Turbo | ✅ |
| Grok-2 | Grok-2 | ✅ |
| MiniMax M2 | MiniMax-M2 | ✅ |
| BailingMoe2 | Ling, Ring | ✅ |
| Ernie 4.5 MoE | ERNIE-4.5 | ✅ |
| OpenAI MoE | GPT-OSS | ✅ |
| Mimo2 | Mimo2 | ✅ |

### Как работают MOE модели

MOE модель имеет **experts** (экспертов) в слоях FFN. Каждый токен активирует только **2-5% экспертов**, остальные не используются. Это ключевое отличие от dense моделей.

**Структура MOE слоя:**
```
Input → Router (top-k) → Active Experts → Output
         ↑                    ↑
    2-5% активных        Sparse matrix multiply
```

**Экспертные тензоры** (то что идёт в "медленную" память):
- `ffn_up_exps.weight` —專家权重矩阵
- `ffn_gate_exps.weight` —专家门控矩阵
- `ffn_down_exps.weight` —专家下投影矩阵
- `ffn_gate_up_exps.weight` — fused up+gate

**Не-экспертные тензоры** (то что всегда в GPU):
- `attn_q/k/v/out.weight` — attention
- `ffn_up/gate/down.weight` — dense FFN (первые слои)
- `token_embd.weight`, `output.weight`

### Стратегии offloading для MOE

#### Стратегия 1: Все слои в GPU (`-ngl 999`)

```bash
llama-server -m model.gguf -ngl 999
```

- Все эксперты в VRAM
- Максимальная скорость
- Требует много VRAM (experts = большие тензоры)
- **Проблема**: 70B MOE = ~140GB → не влезет в 24GB

#### Стратегия 2: Эксперты в CPU (`--n-cpu-moe N`)

```bash
# Первые 20 слоёв: эксперты в CPU, остальные в GPU
llama-server -m model.gguf -ngl 999 --n-cpu-moe 20
```

- Эксперты первых N слоёв → CPU RAM
- Attention + dense FFN → GPU VRAM
- **Эксперты sparse (2-5% активных)** → CPU-GPU transfer минимален
- **Типичный баланс**: `--n-cpu-moe 20` для 70B MOE на 24GB VRAM

#### Стратегия 3: Автоматический (`--fit`)

```bash
llama-server -m model.gguf --fit --fit-margin 2048
```

- Автоматически загружает столько слоёв, сколько влезает в VRAM
- **Нельзя комбинировать** с `--n-cpu-moe` или `-ot`
- `--fit-margin` — запас VRAM в MB (default 1024)

#### Стратегия 4: Точный контроль (`-ot`)

```bash
# Все эксперты → CPU
llama-server -m model.gguf -ngl 999 -ot ".ffn_.*_exps.=CPU"

# Только 특정ные слои → CPU
llama-server -m model.gguf -ngl 999 -ot "blk.(?:0-9|1[0-9]).ffn_.*_exps.=CPU"
```

- Regex-based tensor override
- Максимальный контроль
- Можно комбинировать с `-ngl 999`

### MOE-специальные флаги

| Флаг | Значение | Когда использовать |
|------|----------|-------------------|
| `-fmoe` (default: ON) | Fused up+gate ядро | Всегда (significant speedup) |
| `-ngl 999` | Все слои в GPU | Модель влезает в VRAM |
| `--n-cpu-moe N` | Эксперты первых N слоёв → CPU | MOE > VRAM capacity |
| `--fit` | Авто offload | Большие MOE модели |
| `-ot "regex=CPU"` | Точный контроль тензоров | Продвинутые сценарии |
| `--defer-experts` | Ленивая загрузка экспертов | Медленный SSD, cold-start |
| `--no-offload-only-active` | Все эксперты в GPU | Когда все эксперты активны |
| `-sas` | Async graph eval | Multi-GPU split-mode graph |

### MOE Prefetch (Linux)

```
Флаг: (автоматически при mmap + MOE)
```

Система prefetch proactively загружает страницы экспертных тензоров в page cache:
- Использует `MADV_POPULATE_READ` для предзагрузки
- Multi-threaded worker pool (2MB chunks)
- `MADV_COLD` для освобождения страниц после prompt processing
- **Эффективно для NVMe/SSD**, где latency random read высокое

### Формула памяти для MOE моделей

```
VRAM = Attention Weights + Dense FFN Weights + Active Expert Transfer + KV Cache + Buffers

Для MOE с N layers, E experts, top-k active:
  Attention per layer = 4 × n_heads × head_dim × dtype_size
  Dense FFN per layer = 2 × hidden_dim × intermediate_dim × dtype_size
  Expert per layer = E × 3 × hidden_dim × (intermediate_dim/E) × dtype_size
  Active transfer = top_k/E × Expert per layer (при CPU offload)

Пример: DeepSeek-V3 (671B, 61 experts, top-8):
  Expert weights total: ~500GB
  Active per token: 8/61 × 500GB ≈ 65GB
  At IQ4_KS (4.25 bpw): ~14GB per token's experts
```

## Управление памятью: полный справочник

### mmap (default: ON)

```
Флаги: --mmap (default) / --no-mmap
```

| Режим | Поведение | Плюсы | Минусы |
|-------|-----------|-------|--------|
| mmap (default) | Lazy loading страниц по demand | Быстрый старт, shared pages, экономия RAM | Page faults при first access |
| `--no-mmap` | Eager read в RAM | Нет page faults, предсказуемая latency | Медленный старт, больше RAM |

**Когда использовать `--no-mmap`:**
- Реалтайм требования (нет page faults)
- `-rtr` (run-time repack) — требует модификации данных
- `-mqkv` (merge QKV) — требует модификации данных
- `--no-mmap` + `--mlock` = максимальная предсказуемость

### mlock (default: OFF)

```
Флаги: --mlock / -mlock
```

| Режим | Поведение |
|-------|-----------|
| OFF (default) | ОС может свопить модель на диск |
| ON | `mlock()` на всех страницах модели, never swapped |

**Эффект:**
- Убирает page-in latency при inference
- Модель **всегда** занимает физическую RAM
- **Нельзя** с `--defer-experts` (отключает deferral)
- Уменьшает доступную RAM для других процессов

**Рекомендация для RTX 3090 + 9950X:**
```bash
# Для максимальной производительности (есть достаточно RAM):
--mlock --no-mmap

# Для баланса (default):
# (ничего не нужно, mmap + no-mlock работает хорошо)
```

### n-gpu-layers / -ngl

```
Флаги: -ngl N / --n-gpu-layers N
```

| Значение | Поведение |
|----------|-----------|
| 0 | Чисто CPU inference |
| N (меньше чем слоёв) | Первые N слоёв в GPU |
| 999 (больше чем слоёв) | Все слои в GPU |

**Влияние на скорость:**
```
-ngl 0 (CPU only):     PP ~300 t/s, TG ~15 t/s (на 7B)
-ngl 20 (partial):     PP ~800 t/s, TG ~40 t/s
-ngl 999 (full GPU):   PP ~2000 t/s, TG ~80 t/s  ← ЦЕЛЬ
```

**Для MOE:** `-ngl 999` + `--n-cpu-moe N` —专家可以 быть в CPU, attention в GPU.

### split-mode / -sm

```
Флаги: -sm none|layer|graph|attn
```

| Режим | Описание | Когда использовать |
|-------|----------|-------------------|
| `none` | Одна GPU | 1 GPU, simplest |
| `layer` (default) | Слои распределены по GPU | Multi-GPU, general |
| `graph` | Тензоры разделены по GPU | MOE, large models |
| `attn` | Attention разделён по GPU | Attention-bound models |

**Для MOE на 1 GPU:** `-sm none` (default)
**Для MOE на 2+ GPU:** `-sm graph` (если архитектура поддерживается)

### --fit (автоматический offload)

```
Флаги: --fit / --fit-margin N
```

Автоматически загружает столько слоёв в GPU, сколько влезает в VRAM.

```bash
# Автоматический выбор с запасом 2GB
llama-server -m model.gguf --fit --fit-margin 2048

# Нельзя комбинировать с:
# --n-cpu-moe, -ot (ручные override)
```

**Как работает:**
1. Считает размер каждого слоя + KV cache
2. Жадно назначает слои в GPU пока VRAM не кончится
3. Оставляет `--fit-margin` MB запаса

### --override-tensor / -ot

```
Флаги: -ot "regex=backend"
```

Regex-based контроль размещения тензоров:

```bash
# Все эксперты → CPU
-ot ".ffn_.*_exps.=CPU"

# Определённые слои → CPU
-ot "blk.(?:0|1|2).ffn_.*_exps.=CPU"

# Attention → GPU0, эксперты → CPU
-ot ".attn_.*=GPU0" -ot ".ffn_.*_exps.=CPU"
```

**Backend:** `CPU`, `GPU0`, `GPU1`, и т.д.

### --no-kv-offload / -nkvo

```
Флаги: -nkvo / --no-kv-offload
```

KV cache **всегда** в CPU RAM, даже если модель в GPU.

**Эффект:**
- Экономит VRAM (KV cache может быть огромным)
- Замедляет prompt processing (CPU-GPU transfer KV)
- **Полезно для:** контекст >32K на 24GB VRAM

### KV Cache квантизация — АППАРАТНОЕ УСКОРЕНИЕ

```
Флаги: -ctk TYPE / -ctv TYPE / --k-cache-hadamard / --v-cache-hadamard
```

**АППАРАТНОЕ УСКОРЕНИЕ — ГЛАВНЫЙ КРИТЕРИЙ ДЛЯ KV CACHE:**

| Тип | bpw | CPU AVX-512 | CPU VNNI | GPU dp4a | GPU INT8 TC | Качество |
|-----|-----|:-----------:|:--------:|:--------:|:-----------:|----------|
| **q8_0** | 8 | ✅ R8 repack | косвенно | ✅ dp4a | ❌ | Почти без потерь |
| **q6_0** | 6 | ✅ | ❌ | ✅ dp4a | ❌ | Малые потери |
| **q4_0** | 4 | ✅ nibble unpack | косвенно | ✅ dp4a | ❌ | Заметные потери |
| **q4_1** | 4.5 | ✅ | ❌ | ✅ dp4a | ❌ | Заметные потери |
| **iq4_nl** | 4.5 | ✅ LUT | ❌ | ✅ dp4a | ❌ | Заметные потери |
| **q8_KV** | 8 | ✅ **САМЫЙ БЫСТРЫЙ** | ❌ | ❌ **НЕТ ЯДРА** | ❌ | Почти без потерь |
| **f16** | 16 | ✅ | N/A | ✅ FP16 TC | ❌ | Идеальное |
| q5_0 | 5 | ✅ | ❌ | ❌ **НЕТ FA** | ❌ | ❌ не используй |
| q5_1 | 5.5 | ✅ | ❌ | ❌ **НЕТ FA** | ❌ | ❌ не используй |

**Ключевые факты:**
- `q8_0` — **ЕДИНСТВЕННЫЙ** тип, аппаратно ускорен и на CPU, и на GPU
- `q8_KV` — самый быстрый на CPU, но **НЕТ GPU ядра** (только для CPU-only моделей)
- `q5_0`/`q5_1` — **НЕТ FA ядер** (не используй для KV cache)
- `iq4_nl` — аппаратно ускорен, но качество ниже q4_0

```bash
# ОПТИМАЛЬНО: q8_0 (аппаратно ускорен на CPU + GPU)
llama-server -m model.gguf -ngl 999 -fa \
    -ctk q8_0 -ctv q8_0

# Агрессивный кеш (q4_0 + Hadamard)
llama-server -m model.gguf -ngl 999 -fa \
    -ctk q4_0 -ctv q4_0 \
    --k-cache-hadamard --v-cache-hadamard

# CPU-only модель: q8_KV (самый быстрый на CPU)
llama-server -m model.gguf -ngl 0 -fa \
    -ctk q8_KV -ctv q8_KV

# Per-layer: важные слои q8_0, остальные q4_0
llama-server -m model.gguf -ngl 999 -fa \
    -ctk-first q8_0,8 -ctk-last q4_0,999
```

**Важно:** FA (`--flash-attn`) **обязателен** для квантизованного V cache.

### flash-attn / -fa

```
Флаги: -fa (default: ON) / -no-fa
```

| Режим | Память | Скорость | Качество |
|-------|--------|----------|----------|
| ON (default) | Меньше (нет полной attention матрицы) | Быстрее (CUDA kernels) | Идентичное |
| OFF | Больше | Медленнее | Идентичное |

**Когда ОТКЛЮЧАТЬ:**
- Grok модель (принудительно off)
- LoRA (ошибка при включении)
- OpenPangu (принудительно off)

### NUMA

```
Флаги: --numa distribute|isolate|numactl
```

| Режим | Описание |
|-------|----------|
| `distribute` | Потоки распределены по всем NUMA нодам |
| `isolate` | Потоки только на локальной NUMA ноде |
| `numactl` | Внешний `numactl` контролирует affinity |

**Для 9950X (1 socket):** NUMA не нужен (1 нода)
**Для 2+ socket серверов:** `--numa distribute` или `--numa isolate`

### Размер контекста и память

```
Флаг: -c N / --context N
```

**Формула KV cache:**
```
Standard MHA:
  KV_size = 2 × n_layers × n_heads_kv × head_dim × context_len × dtype_size

MLA (DeepSeek, GLM-DSA):
  KV_size = n_layers × (kv_lora_rank + n_rot) × context_len × dtype_size

Примеры (Llama-3.1-8B, 32 layers, 32 heads, 128 head_dim):
  -c 4096,  f16: 2 × 32 × 8 × 128 × 4096 × 2 = 2GB
  -c 4096,  q8_0: 1GB
  -c 16384, f16: 8GB
  -c 16384, q8_0: 4GB
  -c 131072, f16: 64GB ← не влезет в 24GB VRAM
```

### Batch size и память

```
Флаги: -b N / -ub N / -amb N / -wgt N
```

| Флаг | Назначение | Влияние на память |
|------|------------|-------------------|
| `-b N` | Logical batch size | Пропорционально |
| `-ub N` | Physical batch size | Пропорционально |
| `-amb N` | Max attention batch (MB) | Прямо |
| `-wgt N` | Worst-case graph tokens | Прямо |

**Для RTX 3090:** `-b 2048 -ub 512` (default) — хороший баланс.

## Рекомендации для 9950X + RTX 3090

### Dense модели (7B-34B)

```bash
# 7B: максимум скорости (IQ4_KS веса + q8_0 кеш)
llama-server -m Qwen2.5-7B-IQ4_KS.gguf \
    -ngl 999 -fa -c 8192 \
    -ctk q8_0 -ctv q8_0

# 13B: баланс (IQ4_KS веса + q8_0 кеш)
llama-server -m Qwen2.5-14B-IQ4_KS.gguf \
    -ngl 999 -fa -c 8192 \
    -ctk q8_0 -ctv q8_0

# 34B: контекст 16K+ (IQ4_K веса + q6_0 кеш)
llama-server -m Qwen2.5-32B-IQ4_K.gguf \
    --fit --fit-margin 2048 -fa -c 16384 \
    -ctk q6_0 -ctv q6_0

# 7B: максимальный контекст 128K (IQ4_KS веса + q4_0 кеш + Hadamard)
llama-server -m Model-7B-IQ4_KS.gguf \
    -ngl 999 -fa -c 131072 \
    -ctk q4_0 -ctv q4_0 \
    --k-cache-hadamard --v-cache-hadamard
```

### MOE модели

```bash
# DeepSeek-V2-Lite (16B, 2.4B active): полный offload
llama-server -m DeepSeek-V2-Lite-IQ4_KS.gguf \
    -ngl 999 -fa -c 8192 \
    -ctk q8_0 -ctv q8_0

# Qwen3-235B-A22B (235B, 22B active): experts в CPU
llama-server -m Qwen3-235B-A22B-IQ4_KS.gguf \
    -ngl 999 --n-cpu-moe 30 -fa -c 8192 \
    -ctk q8_0 -ctv q6_0

# DeepSeek-R1 (671B, 37B active): авто + контекст
llama-server -m DeepSeek-R1-IQ4_KS.gguf \
    --fit --fit-margin 2048 -fa -c 4096 \
    --defer-experts \
    -ctk q8_0 -ctv q6_0

# Large MOE с контролем экспертов
llama-server -m Model-70B-MoE-IQ4_KS.gguf \
    -ngl 999 -ot ".ffn_.*_exps.=CPU" -fa -c 8192 \
    -ctk q8_0 -ctv q8_0
```

### Cold-start оптимизация (MOE на SSD)

```bash
# Ленивая загрузка экспертов + mlock dense weights
llama-server -m Model.gguf \
    -ngl 999 --n-cpu-moe 40 \
    --defer-experts --mlock -fa -c 4096
```

### Максимальный контекст на 24GB VRAM

```bash
# 7B модель с контекстом 128K (IQ4_KS + q4_0 кеш + Hadamard)
llama-server -m Model-7B-IQ4_KS.gguf \
    -ngl 999 -fa -c 131072 \
    -ctk q4_0 -ctv q4_0 \
    --k-cache-hadamard --v-cache-hadamard \
    -nkvo

# 13B модель с контекстом 64K (IQ4_KS + q6_0 кеш)
llama-server -m Model-13B-IQ4_KS.gguf \
    -ngl 999 -fa -c 65536 \
    -ctk q6_0 -ctv q6_0
```

## Ссылки

- `/home/rasty/Documents/ik_llama.cpp/examples/quantize/quantize.cpp` — все кванты
- `/home/rasty/Documents/ik_llama.cpp/ggml/include/ggml.h:394-489` — enum ggml_type
- `/home/rasty/Documents/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:861-952` — CPU dispatch
- `/home/rasty/Documents/ik_llama.cpp/ggml/src/ggml-cuda/template-instances/` — GPU kernels
- `/home/rasty/Documents/ik_llama.cpp/github-data/discussions/8` — SOTA quants discussion
- `/home/rasty/Documents/ik_llama.cpp/docs/parameters.md` — полная документация параметров
- `/home/rasty/Documents/ik_llama.cpp/src/llama-load-tensors.cpp:263-335` — tensor loading/overrides
- `/home/rasty/Documents/ik_llama.cpp/src/llama.cpp:3567-3798` — memory estimation
- `/home/rasty/Documents/ik_llama.cpp/ggml/src/ggml-moe-prefetch.cpp` — MOE prefetch system
