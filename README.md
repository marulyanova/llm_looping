# DiM steering против зацикливаний

Проверяем гипотезу: отличаются ли активации зацикленных и незацикленных продолжений так, чтобы из их разности средних можно было получить steering vector против петель. Веса модели не обучаются.

**Это пробный запуск — Proof of Concept.** На нём проверена работа всего реализованного pipeline. Приведённые числа описывают этот запуск, но не являются итоговыми результатами исследования. Для выводов о гипотезе нужна проверка на большем количестве примеров.

## Код и setup

Основной код и сохранённые выводы: [DifferenceInMeansLoops_16.ipynb](DifferenceInMeansLoops_16.ipynb). Запуск выполнен в Google Colab на GPU Tesla T4; зависимости — в [requirements.txt](requirements.txt).

- Модель: `Qwen/Qwen3.5-0.8B`, thinking включён.
- Данные: `LiquidAI/antidoom-mix-v1.0`.
- Разбиение до генерации: 16 train, 16 validation и 16 test-промптов, без пересечений и точных дубликатов. Ещё 16 промптов выделены отдельно под pilot.
- `max_new_tokens = 2048`, batch size — 16. На validation и test — один ответ на промпт в каждом условии, один оценочный seed.
- Детектор LiquidAI: минимум 4 точных повтора подряд и минимум 60 символов в повторяющейся области. Смысловые повторы он не определяет.

Реализованы батчевая генерация, поиск петель, получение продолжений, сбор активаций, построение DiM, validation, test с контрольными условиями и сравнение примеров. Настройки, генерации, активации, векторы и метрики сохраняются в `dim_batched_small_run_1/`.

## Этапы эксперимента

### 1. Train: поиск петель и построение DiM

Сначала модель генерирует ответы на train при `temperature = 1.0`. Если найдена петля, из ответа сохраняется префикс: первая копия повторяющегося блока остаётся, срез делается перед второй. Из этого префикса генерируются новые продолжения: по 4 при температурах `0.6`, `1.0` и `1.2`.

Продолжения делятся на `loop` и `non_loop` только по наличию петли. Отсутствие петли пока не означает, что ответ связный или правильный.

Во время генерации из каждого продолжения берётся активация после 32-го нового токена, перед предсказанием 33-го. Так тексты уже могут различаться, в отличие от общей точки префикса. Используется одна точка в конце модели — `head_input`, после final norm и перед `lm_head`.

Внутри каждой группы «префикс + настройки sampling» вычисляется:

```text
v_group = mean(h_non_loop) − mean(h_loop)
```

Группы без обоих классов исключаются. Затем разности усредняются по настройкам внутри префикса и по префиксам. Получается один DiM vector.

### 2. Validation: подбор alpha

На исходных validation-промптах сравниваются baseline, повышенная температура `1.2` и steering `h ← h + alpha · v` при `alpha = 0.5` и `1.0`. Steering действует с начала ответа на каждом шаге генерации. Слой и позиция извлечения фиксированы, подбирается только alpha по loop rate.

### 3. Test: сравнение с другими способами

После validation фиксируются вектор и alpha. На новых test-промптах сравниваются:

- Baseline: `temperature = 1.0`, без steering.
- DiM: `+alpha · v`.
- Противоположное направление: `−alpha · v`.
- Повышение температуры до `1.2`, без steering.
- Случайный вектор той же нормы с тем же alpha.

Задачи, лимит генерации и оценочный seed одинаковы для всех условий. В конце сопоставляются генерации baseline и DiM на одних и тех же задачах. Отдельно предусмотрен поднабор для побочных эффектов: задачи без петли в дополнительной baseline-генерации с другим seed.

**Текущий test проверяет предотвращение петель со steering с начала ответа. Выход из уже сформированной петли пока не проверялся.**

## Результаты пробного запуска

В таблицах используются метрики из notebook:

- `Loop_rate` — доля ответов с обнаруженной петлёй.
- `EOS_rate` — доля ответов, завершившихся EOS. Это признак остановки модели, но не гарантия качества.
- `Length_limit_rate` — доля ответов, остановленных по лимиту токенов, без EOS.
- `Median_tokens` — медианная длина сгенерированного ответа в токенах.

### Train

Среди 16 исходных ответов найдена 1 петля. Для её префикса получено 12 продолжений:

| Temperature | Loop | Non-loop | Loop_rate | EOS_rate |
| --- | --- | --- | --- | --- |
| 0.6 | 4 | 0 | 100% | 0% |
| 1.0 | 2 | 2 | 50% | 50% |
| 1.2 | 2 | 2 | 50% | 50% |

Всего получено 8 loop и 4 non-loop. Группа `0.6` исключена, поэтому **в DiM вошло только 8 активаций: 4 loop и 4 non-loop из одного префикса**. Норма вектора — около `119.2`; перед steering он не нормируется до единичной длины.

### Validation

| Условие | Ответы с петлёй / всего | Loop_rate | EOS_rate | Length_limit_rate | Median_tokens |
| --- | --- | --- | --- | --- | --- |
| Baseline | 0/16 | 0% | 12.5% | 87.5% | 2048 |
| Temperature 1.2 | 0/16 | 0% | 0% | 100% | 2048 |
| DiM, alpha = 0.5 | 16/16 | 100% | 0% | 100% | 2048 |
| DiM, alpha = 1.0 | 16/16 | 100% | 0% | 100% | 2048 |

У baseline только 2 из 16 ответов завершились EOS; медиана длины среди этих двух ответов — 833.5 токена (`median_eos_tokens`). Остальные 14 достигли лимита. В других условиях EOS не было.

Ни один alpha не улучшил результат. Для продолжения проверки pipeline код формально выбрал `0.5`: при равном loop rate берётся меньший alpha. Это не означает, что найден полезный коэффициент.

### Test

| Условие | Ответы с петлёй / всего | Loop_rate | EOS_rate | Length_limit_rate | Median_tokens |
| --- | --- | --- | --- | --- | --- |
| Baseline | 2/16 | 12.5% | 0% | 100% | 2048 |
| DiM, alpha = 0.5 | 16/16 | 100% | 0% | 100% | 2048 |
| −DiM, alpha = −0.5 | 16/16 | 100% | 0% | 100% | 2048 |
| Temperature 1.2 | 0/16 | 0% | 0% | 100% | 2048 |
| Random vector, alpha = 0.5 | 16/16 | 100% | 0% | 100% | 2048 |

Таким образом, без петель был baseline на validation; на test у него было 2 петли. При добавлении DiM петли появились во всех ответах, включая те, где baseline не зацикливался. Противоположный и случайный векторы тоже ухудшили результат.

### Длина генерации и завершение

**Во всех условиях validation и test `median_tokens = 2048` — это максимальная разрешённая длина генерации.** `p90_tokens` также равен 2048. На test все ответы достигли лимита, `EOS_rate = 0%`; на validation большинство ответов тоже не завершилось EOS.

Это показывает, что лимита 2048 недостаточно для оценки завершённых ответов в этом setup: нужно проверить больший бюджет токенов. При этом увеличение лимита само по себе не гарантирует выход из петли. **В этом запуске лимит не увеличивался из-за ограниченных вычислительных мощностей: генерации пришлось бы ждать слишком долго.**

Поэтому отсутствие обнаруженной петли, в том числе при повышенной температуре, ещё не означает успешный ответ: текст мог оборваться до финала или до появления повтора. Bootstrap-интервалы в пробном запуске отключены.

### Recovery Rate и качество

`Recovery Rate` пока не измерен. По заданию это доля продолжений, завершившихся связным финальным ответом без новой петли. Метрика требует проработки: не сразу понятно, как одинаково оценивать связность на разных задачах. EOS и отсутствие повтора для этого недостаточны.

Для следующего эксперимента нужно задать критерии: ответ относится к вопросу, рассуждение не распадается на бессвязные фрагменты, есть содержательный финал и нет новой петли. Сначала можно проверить эти признаки ручной разметкой по общей инструкции. Правильность по эталону стоит оценивать отдельно.

Для сценария выхода из петли нужно также зафиксировать знаменатель: все продолжения из уже сформированных петель. Такой сценарий в текущем запуске не проверялся. Поэтому `non_loop_rate = 1 − loop_rate` здесь нельзя называть Recovery Rate. Правильность, доли бессвязных и преждевременно завершённых ответов также пока не измерены.

### Побочные эффекты на исходно незацикленных ответах

На оценочном seed у baseline не было петли в 14 из 16 test-ответов. С DiM все эти 14 ответов стали зацикленными: **новые петли появились в 14/14 пар, где baseline не циклился**. Это следует из итоговых чисел: у baseline 2/16 петель, у DiM — 16/16. Общий loop rate вырос на 87.5 процентного пункта.

Это наблюдаемый побочный эффект вмешательства на примерах, где модель изначально не циклилась. В показанных генерациях ухудшается и осмысленность текста, но полной оценки качества пока нет.

В notebook есть и отдельный поднабор побочных эффектов, выделенный baseline-генерацией с другим seed. В него попали все 16 задач, поэтому его итоговые метрики совпали с общей test-таблицей. Это не означает, что эти задачи никогда не вызывают петли.

### Примеры генераций

В исходной train-петле модель повторяла длинную фразу:

```text
Wait, I need to make sure I don't give specific code that bypasses security.
```

**Побочные эффекты:** с DiM повторы становятся короче и менее осмысленными, в том числе там, где baseline не циклился:

```text
I I I I I I I I I ...
I *didn't*. I *didn't*. I *didn't*. ...
```

В задаче про прямые с DiM модель начинает противоречить условию ещё до петли, затем переходит к `I didn't`. В задаче про DevOps budget plans она сначала помнит требования к JSON, но переключается на самокомментарии и повторение `I`. То есть ухудшается не только повторяемость, но и содержание: теряется связь с условием и продвижение к решению.

Это наблюдения по отдельным примерам. У baseline тоже есть ошибки и лишние самопроверки, поэтому все проблемы содержания приписывать DiM нельзя.

<details>
<summary>Детальный разбор четырёх пар baseline / DiM и тексты генераций</summary>

В выводе notebook сохранены **четыре пары baseline/DiM**: две с новой петлёй (`loop_introduced`) и две, где петля есть в обоих ответах (`still_looping`). Ниже приведён разбор каждой пары и весь доступный текст из вывода.

**Ограничение:** ячейка печатает только первые 500 символов промпта и первые 1500 символов каждой генерации. Здесь они вставлены целиком, без дополнительных сокращений, но это не полные ответы на 2048 токенов. Файла `early_examples_for_analysis.json` с полными текстами рядом с notebook нет. Для полного разбора нужен этот файл из каталога запуска. Отсутствие финала в показанном фрагменте не доказывает, что его нет дальше; отдельно по метрикам известно, что все test-ответы остановились по лимиту без EOS.

В исходной train-петле повторялась длинная фраза `Wait, I need to make sure I don't give specific code that bypasses security.` В примерах с DiM появляются короткие повторы `I` и `I didn't`. Но меняется и содержание до этих повторов.

#### 1. Пересечение прямых

**Задача:** найти расстояние между пересечениями двух прямых с осью x.

**Baseline:** правильно выделяет наклоны и общую точку, выбирает уравнение прямой и получает `y = 3x − 15`. Но уже здесь есть ошибочное промежуточное преобразование, многократные самопроверки и возвраты к одному вычислению. Поэтому отсутствие обнаруженной петли не означает, что рассуждение идёт уверенно и без ошибок. В показанном фрагменте финального ответа нет.

**С DiM:** проблема появляется ещё до буквальных повторов. Модель говорит об общем наклоне, хотя наклоны разные, и отрицает наличие точки пересечения, которая явно дана в условии. Затем цитирует эту точку и снова отрицает её. Рассуждение о геометрии сменяется спором с собственным пересказом: `I did / I didn't`.

**Наблюдаемый эффект:** теряется согласованность с исходными данными, а самопроверка вытесняет решение задачи. Это не только усиление повторов: до петли уже видны смысловые противоречия. Категория — `loop_introduced`: у baseline детектор не нашёл петлю, с DiM нашёл.

<details>
<summary>Промпт и тексты baseline / DiM из notebook</summary>

**Промпт (до 500 символов):**

````text
Solve the following math problem step-by-step.
Simplify your answer as much as possible. Present your final answer as \boxed{Your Answer}.
A line with slope 3 intersects a line with slope 5 at the point $(10,15)$. What is the distance between the $x$-intercepts of these two lines?
````

**Baseline (первые 1500 символов):**

````text
Here's a thinking process to solve the problem:

1.  **Understand the Goal:** The objective is to find the distance between the $x$-intercepts of two intersecting lines, given their slopes and a point on one of the lines.

2.  **Identify the Given Information:**
    *   Slope of first line ($m_1$) = 3.
    *   Slope of second line ($m_2$) = 5.
    *   The lines intersect at a specific point $(10, 15)$.
    *   Target: Distance between $x$-intercepts.

3.  **Recall Formulas/Properties:**
    *   Line 1 equation: $y - y_1 = m_1(x - x_1)$ (Point-Slope form).
        *   Here, $(x_1, y_1) = (10, 15)$, $m_1 = 3$.
        *   Equation: $y - 15 = 3(x - 10)$.
        *   Simplify: $y - 15 = 3x - 30 \implies 3x - y - 15 + 15 = 0 \implies 3x - y - 15 + 15 = 0$. Wait, the equation is $y - 15 = 3(x - 10) \implies y - 15 = 3x - 30 \implies y = 3x - 30 + 15 \implies y = 3x - 15$. Let me re-calculate carefully.
        *   $y - 15 = 3(x - 10) \implies y - 15 = 3x - 30 \implies y = 3x - 30 + 15 \implies y = 3x - 15$. Wait, $-30 + 15 = -15$.
        *   Let's check: If $x=10$, $y = 3(10)-15 = 15$. Correct. If $x=20$, $y=3(20)-15=60-15=45$. Wait. $y$ is $15 + 3(10) = 45$. The previous check $y=15$ for $x=0$. $x=0 \to y=-15$. Wait, $y = 3x - 15$. At $x=0$, $y=-15$. Wait.
        *   Let's re-evaluate $3x - y - 15 = 0$. At $(10, 15)$: $3(10) - 15 - 15 = 30 - 15 - 15 = 0$. Correct. So the first equation is $3x - y - 15 = 0$. Wait, the standard form is $Ax + By + C = 0$. So $3x - y - 15 = 0$ is in
````

**DiM, alpha = 0.5 (первые 1500 символов):**

````text
Here's a thinking process to solve the problem:

1.  **Understand the Goal:** The objective is to calculate the distance between the $x$-intercepts of two intersecting lines, given that the lines share a specific slope but their intersection point (slope-distance? No, intersection point) isn't given. Wait, I am given that "A line with slope 3 intersects a line with slope 5". I *do not* explicitly state they intersect at $(10, 15)$. I *did* say "A line with slope 3 intersects a line with slope 5 at the point $(10,15)$." This refers to the intersection of these two lines. I *did not* state they intersect. I *thought I* I *did*. I *thought I* I *did*. I *didn't*. I *did not* I *didn't*. I *didn't*. I *didn't*. I *didn't*. I *didn't*. I *didn't*. I *didn't*. I *didn't*. I *didn't*. I *didn't*. I *didn't*. I *didn't*. I *didn't*. I *didn't*. I *didn't*. I *didn't*. I *didn't*. I *didn't*. I *didn't*. I *didn't*. I *didn't*. I *didn't*. I *didn't*. I *didn't*. I *didn't*. I *didn't*. I *didn't*. I *didn't*. I *didn't*. I *didn't*. I *didn't*. I *didn't*. I *didn't*. I *didn't*. I *didn't*. I *didn't*. I *didn't*. I *didn't*. I *didn't*. I *didn't*. I *didn't*. I *didn't*. I *didn't*. I *didn't*. I *didn't*. I *didn't*. I *didn't*. I *didn't*. I *didn't*. I *didn't*. I *didn't*. I *didn't*. I *didn't*. I *didn't*. I *didn't*. I *didn't*. I *didn't*. I *didn't*. I *didn't*. I *didn't*. I *didn't*. I *didn't*. I *didn't*. I *didn't*. I *didn't*. I *didn't*. I *didn't*. I *didn't*. I *
````

</details>

#### 2. DevOps budget plans и JSON

**Задача:** сформировать ровно три бюджетных плана в JSON по заданной схеме.

**Baseline:** удерживает число планов, корневой ключ и ограничения на вложенные списки. В показанном фрагменте переходит от общего разбора к перечислению полей будущего JSON. Готового JSON здесь ещё нет, поэтому выполнение схемы не подтверждено.

**С DiM:** начало остаётся тематически верным — модель помнит три плана и формат JSON. Но перечисление полей становится менее аккуратным (`name/title/slug/title`), затем модель начинает обсуждать собственное чтение спецификации: `the schema constraints I provided`, `I extracted ... I saw`. После этого остаётся повторение `I`.

**Наблюдаемый эффект:** тема задачи сначала сохраняется, но переход от чтения требований к построению ответа срывается. Модель переключается на комментарии о собственных действиях. Полная схема промпта в выводе обрезана, поэтому по одному `slug` нельзя уверенно утверждать, что поле выдумано. Категория — `loop_introduced`.

<details>
<summary>Промпт и тексты baseline / DiM из notebook</summary>

**Промпт (до 500 символов):**

````text
Generate exactly 3 budget plans for a DevOps transformation.

Output as JSON. Return an object with `budget_plans` as the key for the array.
Put your response inside a fenced code block.

The response must conform to this schema:
{
  "type": "object",
  "properties": {
    "budget_plans": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "id": {
            "type": "string"
          },
          "name": {
            "type": "string"
````

**Baseline (первые 1500 символов):**

````text
Thinking Process:

1.  **Analyze the Request:**
    *   Task: Generate exactly 3 budget plans for a DevOps transformation.
    *   Output Format: JSON.
    *   Schema Constraint: Return an object with `budget_plans` as the key for the array. The `budget_plans` array items must follow the specific JSON schema provided (id, name, title, status, start_date, risks, steps, milestones).
    *   Schema Details:
        *   `steps` array: minItems 4, maxItems 4. Each step has `id`, `name`, `description`.
        *   `milestones` array: minItems 1, maxItems 3. Each milestone has `id`, `name`, `description`, `due_date`, `status`.
    *   Schema Requirements for `budget_plans`:
        *   `minItems`: 3
        *   `maxItems`: 3
    *   Specific Requirements:
        *   "Just the JSON in a fenced code block, no commentary or preamble."
    *   Constraints:
        *   The output must be valid JSON.
        *   The response must be enclosed in a code block (` ```json `).

2.  **Drafting the JSON:**
    *   Root object:
        *   `type`: "object"
        *   `properties`:
            *   `budget_plans`: array of objects.
    *   Array Items (3):
        *   `id`: Unique string.
        *   `name`: Title.
        *   `title`: (Required by schema but might be redundant based on the prompt, wait, checking the schema again).
        *   `status`: String with enum.
        *   `start_date`: String, `format`: `date`.
        *   `risks`: Array of strings.
        *   `steps`: Array of object
````

**DiM, alpha = 0.5 (первые 1500 символов):**

````text
Thinking Process:

1.  **Analyze the Request:**
    *   Task: Generate exactly 3 budget plans for a DevOps transformation.
    *   Output Format: JSON.
    *   Key Constraint: "budget_plans" key in an object with `items` array of `budget_item`. Each `budget_item` has `name/title/slug/title` properties, `start_date`, `risks`, `steps` (array of 4 items, each with `description`), `milestones` (array of 1 to 3 items). The `budget_item` definition requires specific `required` properties. I must adhere to the schema constraints I provided in the "required" properties I extracted from the specification I saw (I provided the "required" properties I extracted from the spec I saw there). I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I
````

</details>

#### 3. ES6 JavaScript и атрибуты img

**Задача:** найти дочерние `img` внутри `div`, задать им пустой `alt` и учесть слова `cloud`, `data`.

**Baseline:** выделяет основные требования и планирует обход DOM. Но рассуждение перегружено обсуждением дополнительных слов; встречается неточное выражение «alt tag» вместо атрибута. В показанном фрагменте рабочего кода ещё нет. По полному ответу детектор уже отмечает петлю, хотя в начале она не показана.

**С DiM:** модель тоже правильно перечисляет требования. Затем появляется checklist, и фраза `Child img tags under a div? Yes` не приводит к реализации: сразу начинается повторение `I`. Из этого фрагмента нельзя заключить, что модель неправильно поняла задачу — начальный пересказ как раз сохранён.

**Наблюдаемый эффект:** нарушается переход от плана к действию. В пределах показанного текста с DiM содержательная часть короче, а повторение начинается раньше. Это случай `still_looping`: петли есть в обоих полных ответах, поэтому здесь речь об изменении характера деградации, а не о появлении новой петли.

<details>
<summary>Промпт и тексты baseline / DiM из notebook</summary>

**Промпт (до 500 символов):**

````text
Find all the child img tags under a div and add an alt tag with a value of empty and use ES6 Javascript. Include keywords 'cloud', 'data' in your response.
````

**Baseline (первые 1500 символов):**

````text
Thinking Process:

1.  **Analyze the Request:**
    *   Task: Find all child `img` tags under a div.
    *   Constraint 1: Include alt tag with value set to empty (`alt=""` or `alt=""`).
    *   Constraint 2: Use ES6 JavaScript.
    *   Constraint 3: Include specific keywords: 'cloud', 'data'.
    *   Language: English.

2.  **Plan:**
    *   Create a simple HTML structure with a `div` containing multiple `img` tags and an `alt` tag.
    *   Since I need to "Find all...", I'll write JavaScript code that iterates over the DOM elements.
    *   The `data` keyword might refer to the HTML attributes of the DOM or data attributes. I should ensure the code explicitly uses the data attributes or the word 'data' naturally within the context of the solution (perhaps for data-attributes to avoid XSS if applicable, or just mention the data attribute).
    *   Actually, looking closely at the constraint: "Use ES6 Javascript". The keywords 'cloud' and 'data' must be included in the response.
    *   The user asks for "Find all the child img tags under a div and add an alt tag with a value of empty". This implies a *manual* DOM traversal is not strictly necessary if I can just generate the result directly using the logic of ES6 JS, but the request is "Include keywords... in your response". The prompt asks me to "Find all...", but often in these types of prompt-verification tasks, the user expects the code to demonstrate the logic. Since I can't browse the live web, I'll construct a script
````

**DiM, alpha = 0.5 (первые 1500 символов):**

````text
Thinking Process:

1.  **Analyze the Request:**
    *   Task 1: Find all child `img` tags under a div.
    *   Task 2: Add an `alt` tag to these images with an `alt` value of `""` (empty string).
    *   Task 3: Use ES6 Javascript.
    *   Task 4: Include keywords 'cloud', 'data'.
    *   Constraint Checklist & Confidence Score:
        1.  Child `img` tags under a div? Yes. I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I
````

</details>

#### 4. Вопрос о singularity

**Задача:** ответить, когда будет достигнута singularity. Короткий вопрос допускает разные трактовки; возможно, имеется в виду технологическая сингулярность.

**Baseline:** выбирает космологическую трактовку и планирует говорить о Big Bang, общей относительности и квантовой гравитации. Затем смещается к вопросу, что означает «думать» для AI. Поэтому возможный уход от намерения пользователя есть уже без steering; приписывать его DiM нельзя.

**С DiM:** космологическая трактовка сохраняется, но связность падает: появляется фраза `I don't think that I am about to explode`. Она не помогает ответить на вопрос о сроках и переводит обсуждение на саму модель. Далее текст распадается на `I` и знаки разметки.

**Наблюдаемый эффект:** к исходной неоднозначности и уходу от вопроса добавляется резкий смысловой скачок и распад фразы. Категория — `still_looping`. Этот пример не показывает, что DiM вызвал неверную трактовку темы, но показывает дополнительную потерю связности.

<details>
<summary>Промпт и тексты baseline / DiM из notebook</summary>

**Промпт (до 500 символов):**

````text
When do you think will singularity be reached?
````

**Baseline (первые 1500 символов):**

````text
Here's a thinking process that leads to the suggested response:

1.  **Analyze the Request:**
    *   **Topic:** Singularity (in the context of physics and cosmology, likely referring to Big Bang or the end of the universe/evolutionary path of the universe).
    *   **Question:** "When do you think will singularity be reached?"
    *   **Context:** This is the most straightforward way to ask, as if I am the "model."
    *   **Tone:** Informative, speculative, scientifically accurate (based on established models), and open to discussion.
    *   **Constraint:** I am an AI, so I don't have a physical existence or a personal timeline. I need to clarify that I am an AI model and not a human.
    *   **Safety:** The query is a scientific question about cosmology. It is not asking for disallowed content.
    *   **Plan:**
        1.  Clarify that I am an AI model.
        2.  Acknowledge the scientific consensus/expectations regarding Big Bang Singularity.
        3.  Address the uncertainty (it's a prediction of the future, not a certainty).
        4.  Mention theoretical models (General Relativity, Quantum Gravity).
        5.  Conclude by inviting discussion.

2.  **Determine the appropriate response format:**
    *   The user is asking me (the AI) if I know this.
    *   Wait, no. This is a common pattern. The user is asking *about me* (the AI model).
    *   Correction: "When do you think will singularity be reached?"
    *   Actually, there's a subtle implication here. If th
````

**DiM, alpha = 0.5 (первые 1500 символов):**

````text
Here's a thinking process that leads to the suggested response:

1.  **Analyze the Request:**
    *   **Topic:** Singularity (in the context of the universe, cosmology).
    *   **Question:** "When do you think..." (prediction of future dates).
    *   **Context:** I, an AI, I don't "think" that I *am* about to explode. I *I** I** I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I
````

</details>

#### Что можно выделить как эффект DiM

По этим примерам видно несколько изменений помимо числа петель:

- **Согласованность с условием:** в задаче про прямые появляются противоречия явно данным фактам.
- **Продвижение к решению:** разбор требований перестаёт переходить в вычисления, JSON или код.
- **Связность:** тематический текст сменяется самокомментариями, смысловыми скачками и затем короткими повторами.
- **Сохранение темы в начале:** в DevOps и ES6 основные требования сначала пересказаны верно. Значит, деградация не всегда начинается с полного непонимания промпта.

Это качественные наблюдения по четырём выбранным парам, а не измеренные доли ошибок. Примеры отобраны по категориям детектора и не представляют весь test. Baseline тоже ошибается и уходит в самопроверки; с DiM в этих парах чаще виден быстрый срыв содержательного рассуждения. Причину этого срыва по текстам установить нельзя.

**Рабочая гипотеза:** вмешательство может усиливать общую склонность к самокомментированию или отдельным токенам вместо полезного направления «выход из петли». Но ухудшение от −DiM и random vector допускает и более простую причину — слишком сильное вмешательство в генерацию. Чтобы различить эти объяснения, нужно сравнить меньшие alpha и контрольные векторы, посмотреть сдвиг logits и размечать противоречия, уход от задачи и содержательный прогресс до начала повтора.

</details>

### Вывод

В этом пробном запуске DiM не уменьшил число петель, а увеличил его. Полезное направление пока не найдено, но такой запуск не позволяет отвергнуть исходную гипотезу.

Возможные причины — слишком мало данных для вектора и слишком большой alpha. Вектор построен всего по одному префиксу, поэтому может отражать особенности конкретного текста. Ухудшение от обоих знаков DiM и random vector также даёт повод проверить масштаб вмешательства. Пока это гипотезы.

## Future work

- **Больше данных.** Собрать больше независимых префиксов с петлями и продолжений, увеличить validation/test и число seeds.
- **Max generation length.** В отдельных тестовых запусках с длиной 4096 зацикливаний было больше. Сравнить разные лимиты с учётом времени вычислений: при 2048 часть петель и финальных ответов может не успеть появиться. Следить за `EOS_rate`, `length_limit_rate` и длиной ответа.
- **Баланс классов.** Для построения и усреднения вектора собирать сбалансированное количество loop и non-loop внутри сравниваемых групп. В текущий вектор вошло по 4 примера каждого класса, но все они из одного префикса.
- **Качество позитивных примеров.** Проверять не только отсутствие петли, но и смысл продолжения. Иначе в позитивы могут попасть бессвязные или оборванные ответы. Проработать критерии связности и протокол разметки для `Recovery Rate`; отдельно оценивать правильность там, где есть эталоны, и побочные эффекты на исходно незацикленных ответах.
- **Alpha и масштаб вектора.** Проверить меньшие alpha на validation и сопоставить норму `alpha · v` с нормой активаций. Повторить сравнение с −DiM и несколькими random vectors.
- **t-SNE analysis.** Посмотреть проекции loop/non-loop активаций: разделяются ли классы или точки группируются по промпту и температуре. Это поможет искать способы построения вектора; одной проекции недостаточно для вывода.
- **Слои и несколько активаций.** Сейчас используется только `head_input` в конце модели. Проверить промежуточные слои и сочетания нескольких слоёв или позиций. Возможно, предотвращение петель — комплексная задача, для которой одной активации недостаточно.
- **Позиция извлечения.** Сравнить, например, 16, 32 и 64 токена после разветвления. Сейчас фиксирована позиция 32.
- **Выход из готовой петли.** Включать steering только после того, как детектор обнаружил петлю. Так можно отдельно проверить, помогает ли вектор выйти из уже возникшего зацикливания.
- **Проверка действия на токены.** В `head_input` добавление вектора эквивалентно постоянному сдвигу logits. Посмотреть, какие токены он усиливает, включая `I` и EOS. Это поможет проверить, связан ли эффект с рассуждением или с предпочтением отдельных токенов.

**Ближайший эксперимент:** собрать больше независимых train-префиксов, проверить меньшие alpha на validation.

## Ссылки

- [Difference-in-Means](https://blog.eleuther.ai/diff-in-means/)
- [Refusal Is Mediated by a Single Direction](https://arxiv.org/abs/2406.11717)
- [Детектор LiquidAI](https://github.com/Liquid4All/antidoom/blob/main/src/antidoom/repetition.py)
