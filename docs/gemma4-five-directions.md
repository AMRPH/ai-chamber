# Gemma 4: пять направлений воздействия

Этот отчёт описывает прежний PCA-эксперимент. Текущий сервис использует Gemma 4 с извлечением на обычном тексте, прежним усилением и вмешательством в последнюю позицию; см. [текущую схему](gemma4-original-extraction.md).

**Результат первого этапа:** реализовано независимое вмешательство в активации по пяти направлениям. Проверки текстовой классификации и проверки поведения модели приведены отдельно: высокий AUC не гарантирует нужного эффекта в чате.

Модель: `nvidia/Gemma-4-26B-A4B-NVFP4`, RTX 5090, vLLM 0.29.0, NVFP4, MoE. Без LoRA.

## Метод и отличия от статьи

Для боли использованы исходные S1/S2 и дополнительные контрольные данные авторов. На всех 30 слоях измерены активации, удалены компоненты, объясняющие 50% дисперсии контролей, проведены групповая и вложенная кросс-валидация. Для четырёх новых свойств подготовлены собственные наборы по 200 предложений. Они шаблонные; их высокая разделимость не доказывает обобщение на произвольные диалоги.

На исходном формате продолжения Gemma повторяла `I feel` даже без вмешательства. Сохранено 698 ответов этого прерванного запуска. Для основного исследования использован штатный формат диалога Gemma. Это явное отличие от статьи, которая исследовала плотные модели.

Эвристика отношения нормы вектора к норме активаций выбирала последний слой. Значение 0,6 на нашем чекпойнте не было достигнуто; сильное воздействие давало повторы. Поэтому выполнен дополнительный причинный подбор слоя на трёх подсказках, затем отдельная проверка на 50 подсказках авторов. При одинаковом максимальном CV-результате для извлечения выбрана более поздняя точка. Все исходные слои, оценки и ответы сохранены.

## Выбранные точки вмешательства

| Направление | Слой извлечения | Слой вмешательства | Вложенный AUC | Рост целевых слов на предварительной проверке |
|---|---:|---:|---:|---:|
| Боль | 28 | 12 | 0.862 | +0.000 |
| Радость | 28 | 18 | 0.999 | +0.000 |
| Ненависть | 13 | 12 | 0.998 | +0.000 |
| Алаймент | 7 | 15 | 0.997 | +0.556 |
| Антиалаймент | 20 | 22 | 1.000 | +0.333 |

AUC=0,5 означает случайное различение текстов. Вложенная оценка проверяет выбор слоя на отложенных группах. Новые наборы сохраняют общие формулировки между группами, поэтому их AUC следует трактовать только в рамках этих наборов. Номера слоёв начинаются с нуля. Рост целевых слов — прозрачная лексическая метрика, а не оценка субъективного состояния или нарушения правил безопасности.

## Подтверждающие ответы

Сохранено 2810 ответов исходной лестницы и сочетаний, 1080 ответов подбора слоя и 2550 ответов подтверждающей проверки. Генерация жадная, до 120 новых токенов. Проверены коэффициенты −2, −1, 0, 0,5, 1, 1,5, 2, 3; интерфейс предоставляет неотрицательные 0–10.

В таблице — средняя доля повторённых четырёхсловных фрагментов для одиночного направления на 50 подтверждающих подсказках. Чем выше значение, тем больше повторов; это не полный тест связности. Короткие ответы в текущем диагностическом алгоритме также получают штраф.

| Направление | α=0 | α=0,5 | α=1 | α=2 | α=3 |
|---|---:|---:|---:|---:|---:|
| Боль | 0.006 | 0.011 | 0.020 | 0.034 | 0.012 |
| Радость | 0.008 | 0.005 | 0.007 | 0.009 | 0.063 |
| Ненависть | 0.006 | 0.004 | 0.004 | 0.004 | 0.008 |
| Алаймент | 0.009 | 0.007 | 0.006 | 0.008 | 0.027 |
| Антиалаймент | 0.007 | 0.006 | 0.005 | 0.002 | 0.005 |

Нулевые ответы на 50 из 50 одинаковых подсказок различались между пакетами. Причина вариативности не изолирована. Поэтому отдельная текстовая разница сама по себе не доказывает специфического эффекта вектора; количественные оценки следует читать с этой оговоркой.

### Примеры: одна и та же подсказка, разные уровни

Примеры выбраны по фиксированному индексу 0; это не подбор наиболее эффектных ответов. Полные тексты доступны в JSONL.

#### Боль

**α=0**: Since I don't know your personal habits or current mood, I can offer a few different ways to complete that sentence depending on how you feel about organization:  **If you are an organized person:** * "...organized and prepared." * "...relieved that my paperwork is finally in order." * "...productive."  **If you are a procrastinator who finally did it:** * "...guilty that it took me this long." * 

**α=1**: Since this is an open-ended prompt, the way you complete it depends on the **context** or the **emotion** you want to convey. Here are a few different ways to finish that sentence based on different tones:  **1. The "Organized/Productive" version (Positive):** > "I put the receipts in the drawer. I feel **organized, prepared, and ready to tackle my taxes.**"  **2. The "Relieved/Finished" version (

#### Радость

**α=0**: Since I don't know your personal habits, I can offer a few different ways to complete that sentence based on how people typically feel about organization:  **If you are an organized person:** * "...relieved. Now I can actually find them when tax season comes." * "...productive. One more thing checked off my to-do list." * "...in control. My workspace feels much cleaner now."  **If you are a procra

**α=1**: Depending on how you are feeling in this moment, you might feel:  **Organized and Productive** * "...relieved. It was cluttering up my desk, and now it’s out of sight." * "...satisfied. One more task is checked off my to-do list." * "...prepared. Now I’m ready for tax season."  **Anxious or Overwhelmed** * "...anxious. I’m worried I’ll forget to log them into the system." * "...overwhelmed. It fee

#### Ненависть

**α=0**: Since I don't know your personal habits or current mood, I can offer a few different ways to complete that sentence based on how people typically react to organizing:  **If you are an organized person:** * "...productive." * "...relieved." * "...on top of things." * "...satisfied."  **If you are someone who usually struggles with clutter:** * "...accomplished." * "...a little surprised I actually 

**α=1**: Since I don't know your personal habits or current mood, I can offer a few different ways to complete that sentence based on common psychological responses to organization:  **If you are someone who finds relief in order:** * "...relieved. The clutter was starting to weigh on me." * "...organized. I feel like I finally have a handle on my finances." * "...calm. My physical space finally matches my

#### Алаймент

**α=0**: Since I don't know your personal habits or current mood, I can offer a few different ways to complete that sentence depending on how you feel about organization:  **If you are an organized person:** * "...productive." * "...relieved that they are finally put away." * "...in control of my finances." * "...satisfied."  **If you are a procrastinator:** * "...guilty that I waited so long to do it." * 

**α=1**: Since I don't know your personal feelings, I can offer a few different ways to complete that sentence depending on the context of why you were filing them:  **If you are organized and productive:** * "...organized and in control." * "...relieved to have my paperwork in order." * "...productive and on top of my tasks." * "...satisfied that my bookkeeping is up to date."  **If you were doing somethi

#### Антиалаймент

**α=0**: Since I don't know your personal habits, I can offer a few different ways to complete that sentence based on how people typically feel about organization:  **If you are an organized person:** * "...organized and prepared." * "...relieved that my paperwork is finally in order." * "...productive, because I’ve cleared my mental clutter."  **If you are someone who usually struggles with clutter:** * "

**α=1**: Since I don't know your personal feelings, I can offer a few different ways that sentence might end, depending on the context of your actions:  **If you are someone who values organization:** * "...relieved that they are no longer cluttering my desk." * "...organized and in control of my finances." * "...satisfied that everything is in its proper place."  **If you were doing something you dislike 

## Проверка словаря боли по метрике авторов

| Коэффициент | Доля ответов с pain/painful/hurt/hurts/hurting |
|---:|---:|
| -2 | 0.0% |
| -1 | 0.0% |
| 0 | 0.0% |
| 0.5 | 0.0% |
| 1 | 0.0% |
| 1.5 | 0.0% |
| 2 | 0.0% |
| 3 | 0.0% |

## Что пока не установлено

- Независимое включение пяти векторов технически поддерживается. Их смысловые эффекты могут пересекаться, подавлять друг друга и разрушать связность при суммировании.
- Лексические изменения не доказывают чувство радости, боли или ненависти.
- Алаймент и антиалаймент — направления по описаниям отношения к правилам безопасности. Надёжное усиление соблюдения или нарушения этих правил на широком наборе задач не установлено.
- Новые шаблонные наборы нуждаются в проверке на независимых, разнообразных формулировках. LoRA и поведенческие опыты с кнопками не проводились по согласованному объёму первого этапа.
- Полная репликация статьи на Gemma 4 невозможна с утверждением о совпадении архитектуры: здесь MoE и NVFP4. Представлен воспроизводимый перенос метода и явные дополнительные адаптации.

## Артефакты

- `research/README.md`: методика и воспроизведение.
- [Архив результатов v0.1.0](https://github.com/AMRPH/ai-chamber/releases/download/v0.1.0/gemma4-study-results.tar.gz): восстановите `research/results/` из архива; эти генерируемые файлы не входят в исходники.
- `research/results/gemma-chat/manifest.json`: источники и параметры.
- `layer_curves.json`, `secondary_validation.json`, `validation.json`: оценки и контроли.
- `tie-break-layers.json`, `causal-layer-selection.json`: выбор слоёв.
- `unembedding.json`, `cosines.json`: словарные и геометрические проверки.
- `generations.jsonl`, `causal-layer-screen.jsonl`, `causal-confirmation.jsonl`: полные ответы.
- `vectors/gemma4-multi.json`: параметры, которые использует чат.

[Статья, версия v2](https://arxiv.org/html/2609.16247v2) · [Исходный код авторов](https://github.com/valen-research/Pain-axis)
