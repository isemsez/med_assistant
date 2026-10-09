# Task: write one like-real Russian outpatient chart (амбулаторная карта) + its extract (выписка) — one fictional patient, two documents

You are preparing demo documents for a Russian-language medical reference
assistant used by physicians. You will produce TWO Markdown documents for ONE
fictional patient: (1) the full outpatient chart itself («медицинская карта
амбулаторного больного», ЭМК) and (2) the extract from it («выписка из
амбулаторной карты»). The patients are fictional; the medicine must be
real-world accurate. Document 2 must be literally derivable from Document 1.

## Step 1 — Research (use the browser, cite nothing in the final files)

Study real Russian record structure and style so the output is authentic:

1. Медицинская карта амбулаторного больного (учётная форма 025/у-04 and its
   modern successors): титульный лист (паспортная часть), лист уточнённых
   диагнозов, записи приёмов, результаты обследований, консультации,
   диспансерное наблюдение — what fields each part contains and in what order.
2. «Выписка из медицинской карты» (форма 027/у and successors): how the chart
   condenses into the extract, section order.
3. Real ЭМК layout (ЕМИАС / ЕГИСЗ) and how Russian physicians actually phrase
   entries: concise professional prose, standard abbreviations (ОАК, БХ, ЭхоКГ,
   СКФ, ФВ ЛЖ, ИМТ, ЧСС, АД…), typical workup for the profile below.

## Step 2 — Patient profile (fixed)

**Woman, 74** — paroxysmal atrial fibrillation + arterial hypertension + CKD
stage 3a. Include ECG findings, CHA₂DS₂-VASc score, anticoagulant and
antihypertensives (МНН, doses). You may add plausible minor comorbidities and
medications from your research. Invent a full fictional ФИО.

## Step 3 — Document 1: the амбулаторная карта itself

A full chart as it would look in a real ЭМК export, with:

- Титульный лист: clinic name (generic, e.g. ГБУЗ «Городская поликлиника № …»),
  ФИО, пол, дата рождения (consistent with age 74 in 2025–2026), адрес
  (fictional), полис ОМС / СНИЛС (fictional-looking), участок, дата открытия
  карты.
- Лист уточнённых диагнозов: dates + diagnoses with МКБ-10 codes + physician.
- 3–4 dated записи приёмов (осмотры терапевтом/кардиологом, 2025–2026):
  жалобы, анамнез, объективный статус по системам, план обследования,
  назначения на приеме.
- Результаты обследований с датами и заключениями: ОАК, биохимия (креатинин,
  СКФ, электролиты), ЭКГ (описание пароксизма ФП), ЭхоКГ, суточное
  мониторирование ЭКГ по показаниям.
- Консультация кардиолога с заключением.
- План диспансерного наблюдения / этапный эпикриз.

Realistic fullness: typically **6000–12000 characters**; no hard size cap.
First line: `# Амбулаторная карта: Фамилия Имя Отчество`. Second paragraph:
«Демонстрационная карта, пациентка вымышленная.»

## Step 4 — Document 2: the выписка из амбулаторной карты (same patient)

The condensed extract the physician would hand over. Hard constraints:

- **Hard limit: 8000 characters** including every space, punctuation mark and
  line break (target 5500–7500). Count characters before finalizing; if over,
  condense anamnesis/status prose — never delete a section.
- First line: `# Пациентка: Фамилия Имя Отчество, женщина, 74 года`.
  Second paragraph: «Демонстрационная карта, пациентка вымышленная.»
- Must contain these exact headers: `## Жалобы`, `## Диагнозы`,
  `## Препараты (МНН)`, `## Аллергии`, `## Ключевые показатели` — plus
  additional realistic sections (анамнез заболевания, анамнез жизни,
  объективный статус, обследования, рекомендации/план наблюдения).
- **Derivability**: every fact in the выписка (dates, lab values, doses,
  diagnoses, scores) must appear in Document 1 — no new facts, no
  contradictions.

## Hard requirements for both documents

- Language: **Russian**, professional medical style; diagnoses with МКБ-10
  codes where natural.
- All persons fictional: invented ФИО; no real patients or real doctors;
  generic plausible clinic names.
- Drugs strictly by МНН with dose, form and frequency.
- Lab/instrumental values with units and realistic numbers (pathological where
  the disease implies it); dates within 2025–2026.
- Internal consistency: age, dates, disease duration, values and prescriptions
  must not contradict each other across both documents.

## Output format (strict)

For each document — Document 1 (карта) first, then Document 2 (выписка):

1. One line: `FILENAME: <name>` — Document 1:
   `Фамилия_ИИ.амбулаторная_карта.md`, Document 2: `Фамилия_ИИ.выписка.md`
   (same Фамилия_ИИ in both; Cyrillic surname, underscore, two Cyrillic
   initials, exact suffixes as shown).
2. The complete file content in a single fenced code block. Nothing outside
   the block belongs to the file; no commentary inside the file.
3. Right after the block: `Символов: N` — the exact character count of the
   file content (characters, not words; include newlines).

No extra commentary between or after the documents.

## Final self-check before answering

- Exactly 2 files, one patient, same Фамилия_ИИ in both filenames, all content
  Russian.
- Карта: full structure (титульный лист, лист уточнённых диагнозов, приёмы,
  обследования, консультация, план наблюдения), count reported.
- Выписка: ≤ 8000 chars (counted, reported), five exact headers present,
  disclaimer line present in both documents.
- No real-person data; МНН-only drugs; выписка adds nothing the карта
  doesn't contain.
