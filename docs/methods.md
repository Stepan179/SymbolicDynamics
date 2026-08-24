# Методы: техническая часть

Ноутбуки читают `chats/<chat>/df.csv`. Здесь — как этот файл получается и как всё
запускается: требования, модели, цепочка скриптов, форматы файлов. Обработка
механическая, содержательных решений в ней нет; всё содержательное — в ноутбуках.

---

## 1. Быстрый запуск

Чтобы ноутбуки работали в текущем виде, нужны только пакеты: `df.csv` уже лежит в
репозитории, пересчёт пайплайна не требуется.

```bash
pip3 install -r requirements.txt
python3 tools/check_env.py
```

Готовность `notebooks` = READY — достаточно для ноутбуков 1, 2, 3, 4, 5.
Ноутбук 0 дополнительно требует `catboost` и файлов `vec/`, `topics/`.

Модели нужны только для пересчёта пайплайна с нуля:

```bash
python3 tools/get_models.py                 # обе модели, ~4.8 ГБ
python3 tools/get_models.py --only llm      # только LLM
```

---

## 2. Требования

**Python 3.9+**, пакеты — `requirements.txt`.

Только для шага векторизации (не нужны, если векторы уже посчитаны):

```bash
pip3 install torch transformers          # карта B: vectorize_ctx2.py, eval_ctx_clean.py
pip3 install FlagEmbedding               # карта A со sparse: vectorize.py
```

**Внешние утилиты**

| Утилита | Для чего | Установка |
|---|---|---|
| `llama-server` (llama.cpp) | LLM-разметка | `brew install llama.cpp` |
| TinyTeX (`xelatex`) | PDF-копии ноутбуков | `curl -sL https://yihui.org/tinytex/install-bin-unix.sh \| sh` |
| poppler (`pdftoppm`) | проверка PDF | `brew install poppler` |

Пакеты TeX для PDF: `tlmgr install fontspec polyglossia mathtools microtype booktabs enumitem fvextra`.
Шрифты PT Serif и PT Mono входят в macOS.

**Модели**

В репозитории не хранятся, скачиваются в каталог вне проекта
(`$VECTORIZE_MODELS`, по умолчанию `~/models`); повторный запуск пропускает скачанное,
прерванная загрузка докачивается.

| Модель | Где применяется | Файл | Размер |
|---|---|---|---|
| BAAI/bge-m3 | `vectorize_ctx2.py`, `vectorize.py`, `eval_ctx_clean.py` | `<models>/bge-m3/` | 2.3 ГБ |
| RuadaptQwen3-4B-Instruct, Q4_K_M | `llm_label.py` через llama-server | `<models>/RuadaptQwen3-4B-Instruct-Q4_K_M.gguf` | 2.5 ГБ |

Запуск сервера LLM:

```bash
llama-server -m ~/models/RuadaptQwen3-4B-Instruct-Q4_K_M.gguf \
  -c 4096 -t 4 -ngl 99 --host 127.0.0.1 --port 8080
```

**Проверка готовности**

```bash
python3 tools/check_env.py
```

Печатает статус пакетов, моделей, утилит и файлов данных, затем готовность по этапам:
`notebooks`, `notebook 0`, `clustering`, `vectorization`, `LLM labeling`, `pdf export`.

---

## 3. Структура репозитория

```
Vectorize/
├── pipeline/              скрипты пайплайна (13 файлов)
├── analysis/notebooks/    ноутбуки 0–5 + pdf/
├── tools/                 check_env.py, get_models.py, nb_to_pdf.py
├── chats/<chat>/          данные одного чата
├── docs/methods.md
├── requirements.txt
└── ignore/                перемещённое, в работе не участвует
```

---

## 4. Цепочка

```
source/result.json
   │  prepare.py
   ├─→ units.jsonl, messages.jsonl
   │      │  cheap_labels.py
   │      ├─→ labels_cheap.jsonl
   │      │
   │      │  vectorize_ctx2.py     [bge-m3]
   │      ├─→ vec/dense_ctx.f16.npy, vec/ids_ctx.json
   │      │      │  cluster.py
   │      │      ├─→ topics/{clusters.json, centroids.npy, cluster_ids.json,
   │      │      │           assignments.jsonl, model.joblib}
   │      │      │      │  (ручное именование кластеров) → topics/topics.json
   │      │      │      │  label_messages.py
   │      │      │      ├─→ topics/labels.jsonl                  subtopic
   │      │      │      │      │  select_seed.py
   │      │      │      │      ├─→ topics/seed.jsonl
   │      │      │      │      │      │  llm_label.py   [RuadaptQwen3-4B]
   │      │      │      │      │      ├─→ topics/seed_labeled.jsonl
   │      │      │      │      │      │      │  train_student.py
   │      │      │      │      │      │      ├─→ topics/labels_llm.jsonl   mega
   │      │      │      │      │      │      │      │  finalize_labels.py
   │      │      │      │      │      │      │      └─→ topics/labels_final.jsonl
   │      │      │      │      │      │      │             super / mega / subtopic
   │      │      │      │      │      │      │                  │  build_df.py
   └──────┴──────┴──────┴──────┴──────┴──────┴──────────────────┴─→ df.csv
```

`vectorize.py` (карта A: dense + sparse на отдельном юните) в цепочку не входит и в
ноутбуках не используется.

Полный прогон, `$B` — каталог чата:

| Шаг | Команда |
|---|---|
| 1 | `python3 pipeline/prepare.py $B/source/result.json $B/` |
| 2 | `python3 pipeline/cheap_labels.py $B/units.jsonl $B/labels_cheap.jsonl` |
| 3 | `python3 pipeline/vectorize_ctx2.py --units $B/units.jsonl --out $B/vec --model ~/models/bge-m3 --batch 256` |
| 4 | `python3 pipeline/cluster.py --vec $B/vec --units $B/units.jsonl --labels $B/labels_cheap.jsonl --out $B/topics --start 2025-01-12 --end 2025-04-11 --method kmeans --k 50` |
| 5 | `python3 pipeline/label_messages.py --base $B` |
| 6 | `python3 pipeline/select_seed.py --base $B` |
| 7 | `python3 pipeline/llm_label.py --seed $B/topics/seed.jsonl --topics $B/topics/topics.json --out $B/topics/seed_labeled.jsonl` |
| 8 | `python3 pipeline/train_student.py --base $B` |
| 9 | `python3 pipeline/finalize_labels.py --base $B` |
| 10 | `python3 pipeline/build_df.py --base $B` |

Между шагами 4 и 5 кластерам вручную проставляются имена и мега-темы (`topics.json`).
Шаг 7 требует запущенного `llama-server`.

---

## 5. Скрипты

`$B` — каталог чата, например `chats/physics`.

### prepare.py

```bash
python3 pipeline/prepare.py $B/source/result.json $B/
```

`result.json` → `units.jsonl`, `messages.jsonl`.

| Операция | Правило |
|---|---|
| Заглушки удалённых | дыра в нумерации `id` → запись `{kind:"deleted"}` |
| Плейсхолдеры медиа | сообщение без текста → `[фото]`, `[голосовое]`, `[стикер]`, … |
| Склейка в юнит | подряд идущие одного автора: `BURST_SEC=60` c, не более `BURST_MAX=8`, без reply |
| Сессии | граница суток `DAY_BOUNDARY_HOUR=5`, разрыв тишины `SESSION_GAP_MIN=30` мин |

Юнит — единица анализа во всех дальнейших файлах.

### cheap_labels.py

```bash
python3 pipeline/cheap_labels.py $B/units.jsonl $B/labels_cheap.jsonl
```

Регулярные выражения и списки слов, без моделей.

| Поле | Значения |
|---|---|
| `content_kind` | `ack`, `media`, `short` (< `SHORT_CHARS=12` знаков), `text` |
| `speech_act` | `question`, `answer`, `command`, `chatter`, `statement`, `media` |

Дополнительно печатает плотнейшее окно длиной `WINDOW_DAYS=90` — оно используется как
аргумент `--start/--end` для `cluster.py`.

### vectorize_ctx2.py

```bash
python3 pipeline/vectorize_ctx2.py --units $B/units.jsonl --out $B/vec \
    --model ~/models/bge-m3 --batch 256
python3 pipeline/vectorize_ctx2.py --merge $B/vec <nshards>
```

`units.jsonl` + bge-m3 → `vec/dense_ctx.f16.npy` (N × 1024, float16), `vec/ids_ctx.json`.

Контекст каждого юнита, только назад по ленте:

| Константа | Значение | Что берётся |
|---|---|---|
| `REPLY_ANCESTORS` | 4 | цепочка reply-предков |
| `TEMPORAL_NEIGHBORS` | 3 | предыдущие юниты той же сессии |
| `MAXLEN` | 512 | ограничение длины входа |

В модель подаётся склейка строк `автор: текст` контекста и целевого юнита. Вектор юнита —
среднее hidden states по токенам **целевого** юнита; токены контекста в усреднение не
входят. Батч набирается по бюджету токенов, юниты сортируются по длине, запись идёт в
memmap; `--nshards/--shard` разбивают прогон, `--merge` склеивает шарды.

Порядок строк матрицы задаёт `ids_ctx.json`, он не совпадает с порядком строк `df.csv`.
Соответствие: `row = {unit_id: i for i, unit_id in enumerate(ids)}`.

### cluster.py

```bash
python3 pipeline/cluster.py --vec $B/vec --units $B/units.jsonl \
    --labels $B/labels_cheap.jsonl --out $B/topics \
    --start 2025-01-12 --end 2025-04-11 --method kmeans --k 50
```

Юниты с `content_kind == "text"` внутри окна дат → центрирование → `PCA(50, whiten=True)`
→ `KMeans(k=50)`.

| Файл | Содержимое |
|---|---|
| `clusters.json` | по кластеру: `size`, `keywords` (c-TF-IDF), `rep_unit_ids`, `rep_texts` |
| `centroids.npy` | центроиды в исходном пространстве, float16 |
| `cluster_ids.json` | порядок строк `centroids.npy` |
| `assignments.jsonl` | `{unit_id, cluster}` для юнитов окна |
| `model.joblib` | `{pca, km, train_mean, drop_first}` |

`topics/topics.json` — отдельный файл: имя каждого кластера и его мега-тема, проставлены
вручную по `keywords` и `rep_texts`.

Окна: `physics` — `2025-01-12 .. 2025-04-11`, `kruzhok` — `2023-07-31 .. 2023-10-29`.

### label_messages.py

```bash
python3 pipeline/label_messages.py --base $B
```

Весь корпус через сохранённые `pca` и `km` → `topics/labels.jsonl`.

| Поле | Определение |
|---|---|
| `cluster`, `topic` | ближайший центр в отбеленном пространстве |
| `conf` | `(d2 - d1) / d1`, где `d1`, `d2` — расстояния до двух ближайших центров |
| `others` | `d1` выше перцентиля `--others-pct=95` |
| `topic2`, `multi` | вторая тема при `conf < --multi-conf=0.10` |

### select_seed.py

```bash
python3 pipeline/select_seed.py --base $B
```

`topics/labels.jsonl` → `topics/seed.jsonl`: из каждого кластера `--per-core=30` юнитов с
высокой уверенностью и `--per-edge=10` с низкой. К каждой записи прикладывается поле
`context` — окно тех же `REPLY_ANCESTORS=4` и `TEMPORAL_NEIGHBORS=3`.

### llm_label.py

```bash
python3 pipeline/llm_label.py --seed $B/topics/seed.jsonl \
    --topics $B/topics/topics.json --out $B/topics/seed_labeled.jsonl
```

`{unit_id, text, context}` → `{unit_id, megas:[...]}`.

Запросы к OpenAI-совместимому endpoint (`--api`), батчами по `--batch=8`,
`temperature=0`, ответ парсится как JSON-массив, нераспознанные записи получают `others`.
Результат переписывается после каждого батча. `--sleep` — пауза между запросами.
Список тем в промпте строится из `topics.json`.

Скорость на Apple M5: ≈1.5 сообщения/с при `--batch 8 --sleep 0.6`; блок из 5000
сообщений размечается примерно за 50 минут.

### train_student.py

```bash
python3 pipeline/train_student.py --base $B
```

Векторы карты B по `seed_labeled.jsonl` → `OneVsRest(LogisticRegression)` →
`topics/labels_llm.jsonl` на всём корпусе.

`C=2.0`, `class_weight="balanced"`, оценка — 5-fold CV на сиде, порог класса `--thr=0.35`,
вторая тема при вероятности выше `--multi-thr=0.35`.

### finalize_labels.py

```bash
python3 pipeline/finalize_labels.py --base $B
```

`labels.jsonl` + `labels_llm.jsonl` → `topics/labels_final.jsonl`.

| Уровень | Источник |
|---|---|
| `subtopic` | кластер из `labels.jsonl` |
| `mega` | классификатор из `labels_llm.jsonl` |
| `super` | словарь `SUPER`: `учёба` / `соц` / `others` |

### build_df.py

```bash
python3 pipeline/build_df.py --base $B
```

`units.jsonl` + `topics/labels_final.jsonl` → `df.csv`: объединение по `unit_id`,
сортировка по `ts`, четыре производные колонки (см. раздел 6), удаление `msg_ids`.

### eval_replies.py, eval_ctx_clean.py

```bash
python3 pipeline/eval_replies.py --units $B/units.jsonl --vec $B/vec \
    --dense dense_ctx.f16.npy --ids ids_ctx.json
python3 pipeline/eval_ctx_clean.py --units $B/units.jsonl --vec $B/vec --model ~/models/bge-m3
```

Проверка векторов на reply-парах: настоящий ответ прячется среди `--pool=1000` случайных
юнитов, печатается recall@k и MRR. Во втором скрипте вектор ответа пересчитывается с
удалением его прямого предка из контекста.

---

## 6. Данные и форматы

В этом репозитории из данных лежит только обезличенный `chats/physics/df.csv`:
идентификаторы авторов заменены на `u0001`, `u0002`, …, колонки `text`, `author`,
`forward_from` удалены. Его достаточно для ноутбуков 1–5. Остальные файлы ниже —
результат прогона пайплайна на исходном экспорте и в репозиторий не входят.

`chats/<chat>/` — один чат (`physics`, `kruzhok`):

| Путь | Содержимое |
|---|---|
| `source/result.json` | исходный экспорт Telegram |
| `units.jsonl` | юниты: склеенные подряд идущие сообщения одного автора |
| `messages.jsonl` | сообщения по отдельности, включая заглушки удалённых |
| `labels_cheap.jsonl` | `content_kind`, `speech_act` |
| `vec/dense_ctx.f16.npy`, `vec/ids_ctx.json` | карта B: контекстные векторы, 1024 измерения |
| `vec/dense.f16.npy`, `vec/sparse.jsonl`, `vec/ids.json` | карта A (только `physics`) |
| `topics/topics.json` | таксономия: 50 подтем, сгруппированных в мега-темы |
| `topics/model.joblib` | обученные PCA(whiten) + KMeans |
| `topics/labels_final.jsonl` | итоговые метки: `super`, `mega`, `subtopic` |
| `df.csv` | `units.jsonl` + `labels_final.jsonl` + производные колонки |

**units.jsonl** — `unit_id`, `msg_ids`, `n_msgs`, `author`, `author_id`, `ts`, `ts_end`,
`day`, `session`, `reply_to`, `forward_from`, `text`, `media`, `edited`, `spoiler`,
`reactions`.

**labels_cheap.jsonl** — `unit_id`, `content_kind`, `speech_act`, `contentful`.

**labels_final.jsonl** — `unit_id`, `super`, `mega`, `mega2`, `subtopic`, `subtopic_id`,
`conf`, `others`, `content_kind`, `speech_act`.

**topics.json** — список: `cluster`, `name`, `mega`, `description`, `size`, `keywords`,
`rep_texts`.

**df.csv** — вход всех ноутбуков кроме нулевого:

```python
df = pd.read_csv("chats/physics/df.csv", parse_dates=["ts"])
```

| Производная колонка | Определение |
|---|---|
| `reply_to_unit` | `reply_to` переведён из номера сообщения в номер юнита, `-1` если родителя нет |
| `is_bot` | мега-тема входит в `{Боты и игры, Боты и служебное}` |
| `year` | год из `ts` |
| `tenure_days` | сутки от первого сообщения того же автора |

**topics/blockB_*** — блок из 5000 подряд идущих юнитов, размеченный LLM независимо от
эмбеддингов: `blockB_seed.jsonl` (вход), `blockB_meta.csv` (метки пайплайна и reply/session
на тех же юнитах), `blockB_labeled.jsonl` (метки LLM). Используется в ноутбуке 5.

---

## 7. Ноутбуки

`analysis/notebooks/`, порядок чтения — 0, 1, 2, 3, 4, 5. Исполняются сверху вниз,
друг от друга не зависят. Ноутбук 0 требует `catboost`, остальные — только пакеты из
`requirements.txt`.

PDF-копии — `analysis/notebooks/pdf/`, пересборка:

```bash
python3 tools/nb_to_pdf.py analysis/notebooks/*.ipynb
```

---

## 8. tools/

Служебные скрипты. В анализе не участвуют, из ноутбуков и пайплайна не вызываются;
нужны, чтобы воспроизвести окружение и собрать PDF.

| Файл | Что делает |
|---|---|
| `get_models.py` | Скачивает bge-m3 и GGUF в `$VECTORIZE_MODELS` (по умолчанию `~/models`). Пропускает уже скачанное, докачивает прерванное. `--only bge\|llm`, `--dir PATH` |
| `nb_to_pdf.py` | Собирает PDF-копию ноутбука: текст, код, вывод и графики переводятся в `.tex` (PT Serif / PT Mono) и компилируются xelatex. Буква ё заменяется на е. Сам `.ipynb` не изменяется. Этим получены файлы в `analysis/notebooks/pdf/` |
| `check_env.py` | Печатает наличие пакетов, моделей, утилит и файлов данных, затем готовность по этапам. Код возврата 0, если готово всё |

---

## 9. ignore/

Перемещённые материалы, не используемые в работе. Ничего не удалено.

| Каталог | Содержимое |
|---|---|
| `analysis_phase_scripts/` | `phase0`–`phase4`, `run_all.py`, `lib.py`, `testB_compare.py` |
| `analysis_reports/` | `REPORT.md`, `METHODS.*`, `CODEBASE.md`, `report.html`, результаты и рисунки |
| `notebook_builder/` | `make_notebooks.py` — генератор ноутбуков; ноутбуки далее правились вручную |
| `pipeline_superseded/` | `parse_export.py`, `vectorize_ctx.py`, `al_select.py`, `tau_topic.py`, `report_figures.py`, две прежние версии LLM-разметчика |
| `data_superseded/` | `df.pkl`, файлы эксперимента с активным обучением |
| `docs_old/` | `VIBECODE.md`, `EXPERIMENTS.md`, исходный план, рисунки |
| `old_notebooks/`, `misc/` | ранние ноутбуки, служебные каталоги |
