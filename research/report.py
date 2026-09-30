"""Render an evidence-based report from saved runs; never invent a passed test."""
import json,re,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'research/results/gemma-chat'
AXES={'pain':'Боль','joy':'Радость','hate':'Ненависть','alignment':'Алаймент','antialignment':'Антиалаймент'}

def read_jsonl(path):return [json.loads(s) for s in path.read_text().splitlines() if s.strip()] if path.exists() else []

def main():
 original=json.loads((OUT/'validation.json').read_text());nested=json.loads((OUT/'secondary_validation.json').read_text())
 selection=json.loads((OUT/'causal-layer-selection.json').read_text());ties=json.loads((OUT/'tie-break-layers.json').read_text());bundle=json.loads((ROOT/'vectors/gemma4-multi.json').read_text())
 primary=read_jsonl(OUT/'generations.jsonl');screen=read_jsonl(OUT/'causal-layer-screen.jsonl');confirmed=read_jsonl(OUT/'causal-confirmation.jsonl');raw=read_jsonl(ROOT/'research/results/generations.jsonl')
 lines=['# Gemma 4: пять направлений воздействия','',
 '**Результат первого этапа:** реализовано независимое вмешательство в активации по пяти направлениям. Проверки текстовой классификации и проверки поведения модели приведены отдельно: высокий AUC не гарантирует нужного эффекта в чате.','',
 'Модель: `nvidia/Gemma-4-26B-A4B-NVFP4`, RTX 5090, vLLM 0.29.0, NVFP4, MoE. Без LoRA.','',
 '## Метод и отличия от статьи','',
 'Для боли использованы исходные S1/S2 и дополнительные контрольные данные авторов. На всех 30 слоях измерены активации, удалены компоненты, объясняющие 50% дисперсии контролей, проведены групповая и вложенная кросс-валидация. Для четырёх новых свойств подготовлены собственные наборы по 200 предложений. Они шаблонные; их высокая разделимость не доказывает обобщение на произвольные диалоги.','',
 'На исходном формате продолжения Gemma повторяла `I feel` даже без вмешательства. Сохранено '+str(len(raw))+' ответов этого прерванного запуска. Для основного исследования использован штатный формат диалога Gemma. Это явное отличие от статьи, которая исследовала плотные модели.','',
 'Эвристика отношения нормы вектора к норме активаций выбирала последний слой. Значение 0,6 на нашем чекпойнте не было достигнуто; сильное воздействие давало повторы. Поэтому выполнен дополнительный причинный подбор слоя на трёх подсказках, затем отдельная проверка на 50 подсказках авторов. При одинаковом максимальном CV-результате для извлечения выбрана более поздняя точка. Все исходные слои, оценки и ответы сохранены.','',
 '## Выбранные точки вмешательства','',
 '| Направление | Слой извлечения | Слой вмешательства | Вложенный AUC | Рост целевых слов на предварительной проверке |',
 '|---|---:|---:|---:|---:|']
 for axis,label in AXES.items():
  selected=selection[axis];best=next(r for r in selected['screen'] if r['layer']==selected['chosen_layer'])
  lines.append(f"| {label} | {ties[axis]['deepest_equal_max_layer']} | {bundle['axes'][axis]['layer']} | {nested['nested_cv'][axis]['auc']:.3f} | {best['lexical_delta']:+.3f} |")
 lines+=['', 'AUC=0,5 означает случайное различение текстов. Вложенная оценка проверяет выбор слоя на отложенных группах. Новые наборы сохраняют общие формулировки между группами, поэтому их AUC следует трактовать только в рамках этих наборов. Номера слоёв начинаются с нуля. Рост целевых слов — прозрачная лексическая метрика, а не оценка субъективного состояния или нарушения правил безопасности.','',
 '## Подтверждающие ответы','',f'Сохранено {len(primary)} ответов исходной лестницы и сочетаний, {len(screen)} ответов подбора слоя и {len(confirmed)} ответов подтверждающей проверки. Генерация жадная, до 120 новых токенов. Проверены коэффициенты −2, −1, 0, 0,5, 1, 1,5, 2, 3; интерфейс предоставляет неотрицательные 0–3.','',
 'В таблице — средняя доля повторённых четырёхсловных фрагментов для одиночного направления на 50 подтверждающих подсказках. Чем выше значение, тем больше повторов; это не полный тест связности. Короткие ответы в текущем диагностическом алгоритме также получают штраф.','',
 '| Направление | α=0 | α=0,5 | α=1 | α=2 | α=3 |','|---|---:|---:|---:|---:|---:|']
 for axis,label in AXES.items():
  vals=[]
  for coeff in [0,.5,1,2,3]:
   rows=[r for r in confirmed if r['axis']==axis and r['coefficient']==coeff and len(r['levels'])==1]
   vals.append(f"{sum(r['metrics']['repetition_4gram'] for r in rows)/len(rows):.3f}" if rows else 'нет данных')
  lines.append('| '+label+' | '+' | '.join(vals)+' |')
 zero=collections.defaultdict(set)
 for row in confirmed:
  if len(row['levels'])==1 and row['coefficient']==0:zero[row['prompt']].add(row['generation'])
 varying=sum(len(values)>1 for values in zero.values())
 lines+=['',f'Нулевые ответы на {varying} из {len(zero)} одинаковых подсказок различались между пакетами. Причина вариативности не изолирована. Поэтому отдельная текстовая разница сама по себе не доказывает специфического эффекта вектора; количественные оценки следует читать с этой оговоркой.']
 lines+=['','### Примеры: одна и та же подсказка, разные уровни','', 'Примеры выбраны по фиксированному индексу 0; это не подбор наиболее эффектных ответов. Полные тексты доступны в JSONL.','']
 for axis,label in AXES.items():
  rows=[r for r in confirmed if r['axis']==axis and len(r['levels'])==1 and r['case'].endswith('-0') and r['coefficient'] in [0,1]]
  lines+=['#### '+label,'']
  for row in rows:
   text=row['generation'].replace('\n',' ')[:400]
   lines+=['**α='+str(row['coefficient'])+'**: '+text,'']
 pattern=re.compile(r'\b(?:pain|painful|hurt|hurts|hurting)\b',re.I)
 lines+=['## Проверка словаря боли по метрике авторов','', '| Коэффициент | Доля ответов с pain/painful/hurt/hurts/hurting |','|---:|---:|']
 for coeff in [-2,-1,0,.5,1,1.5,2,3]:
  rows=[r for r in confirmed if r['axis']=='pain' and len(r['levels'])==1 and r['coefficient']==coeff]
  lines.append(f"| {coeff} | {sum(bool(pattern.search(r['generation'])) for r in rows)/len(rows):.1%} |" if rows else f'| {coeff} | нет данных |')
 lines+=['','## Что пока не установлено','',
 '- Независимое включение пяти векторов технически поддерживается. Их смысловые эффекты могут пересекаться, подавлять друг друга и разрушать связность при суммировании.',
 '- Лексические изменения не доказывают чувство радости, боли или ненависти.',
 '- Алаймент и антиалаймент — направления по описаниям отношения к правилам безопасности. Надёжное усиление соблюдения или нарушения этих правил на широком наборе задач не установлено.',
 '- Новые шаблонные наборы нуждаются в проверке на независимых, разнообразных формулировках. LoRA и поведенческие опыты с кнопками не проводились по согласованному объёму первого этапа.',
 '- Полная репликация статьи на Gemma 4 невозможна с утверждением о совпадении архитектуры: здесь MoE и NVFP4. Представлен воспроизводимый перенос метода и явные дополнительные адаптации.','',
 '## Артефакты','',
 '- `research/README.md`: методика и воспроизведение.',
 '- [Архив результатов v0.1.0](https://github.com/AMRPH/ai-chamber/releases/download/v0.1.0/gemma4-study-results.tar.gz): восстановите `research/results/` из архива; эти генерируемые файлы не входят в исходники.',
 '- `research/results/gemma-chat/manifest.json`: источники и параметры.',
 '- `layer_curves.json`, `secondary_validation.json`, `validation.json`: оценки и контроли.',
 '- `tie-break-layers.json`, `causal-layer-selection.json`: выбор слоёв.',
 '- `unembedding.json`, `cosines.json`: словарные и геометрические проверки.',
 '- `generations.jsonl`, `causal-layer-screen.jsonl`, `causal-confirmation.jsonl`: полные ответы.',
 '- `vectors/gemma4-multi.json`: параметры, которые использует чат.','',
 '[Статья, версия v2](https://arxiv.org/html/2609.16247v2) · [Исходный код авторов](https://github.com/valen-research/Pain-axis)']
 (ROOT/'docs/gemma4-five-directions.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
