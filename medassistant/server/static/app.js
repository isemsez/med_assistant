"use strict";

const $ = (sel) => document.querySelector(sel);
const state = { sessions: [], current: null, busy: false, streamActive: false };
let loadPromise = Promise.resolve();
// The in-flight turn: {sid, wrap, question, pdf, done}. loadSession wipes
// #messages on every switch, so the live nodes must be re-attachable —
// while it streams, the store holds at most the question (the assistant
// row with its reasoning lands only when the answer completes).
let activeStream = null;
// Stop control (owner decision 2026-10-15): while the answerer generates,
// #send morphs into «Стоп» (Esc works too); a click POSTs
// /api/sessions/{id}/stop — the server drops the llama-server connection, deletes
// the turn's rows and ends the stream with a `stopped` event, which puts
// the question back into the composer for editing. Answerer phase only.
let stopping = false; // stop clicked for the in-flight turn (one shot)
let sendSeq = 0; // guards composer state when streams overlap (send during appendix)

// esc() lives in util.js, loaded before this script.

function fmt(text) {
  return esc(text)
    .replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>")
    // emphasis * must open before a non-space and close after one (CommonMark
    // edge rule) — otherwise list-bullet markers pair with real emphasis and
    // italicize whole bullet lines
    .replace(/\*(\S(?:[^*\n]*[^\s*])?)\*/g, "<i>$1</i>");
}

function pdfPayload(text) {
  const t = text.trim();
  const sp = t.search(/\s/);
  if (sp === -1) return "";
  const first = t.slice(0, sp).toLowerCase();
  return (first === "pdf" || first === "пдф") ? t.slice(sp).trim() : "";
}

async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r;
}

function insertAfter(node, anchor) {
  (anchor && anchor.parentNode ? anchor.parentNode : $("#messages"))
    .insertBefore(node, anchor ? anchor.nextSibling : null);
}

// ---------- sidebar ----------

async function refreshSessions() {
  const r = await api("/api/sessions");
  state.sessions = await r.json();
  const box = $("#sessions");
  box.innerHTML = "";
  for (const s of state.sessions) {
    const div = document.createElement("div");
    div.className = "session" + (s.id === state.current ? " active" : "");
    div.innerHTML = `${esc(s.title || `Беседа ${s.id}`)}
      <span class="meta">${s.n_messages} сообщ. · ${esc(s.updated_at.slice(0, 16).replace("T", " "))}</span>`;
    div.onclick = () => loadSession(s.id);
    const del = document.createElement("button");
    del.className = "session-del";
    del.textContent = "×";
    del.title = "Удалить беседу";
    del.onclick = (e) => deleteSession(s, e);
    div.appendChild(del);
    box.appendChild(div);
  }
}

async function deleteSession(s, e) {
  e.stopPropagation();
  if (state.streamActive && s.id === state.current) return;
  const title = s.title || `Беседа ${s.id}`;
  try {
    await api(`/api/sessions/${s.id}`, { method: "DELETE" });
  } catch (err) {
    noteNode(`Ошибка удаления беседы: ${err.message}`, true, null);
    return;
  }
  if (s.id === state.current) {
    state.current = null;
    $("#messages").innerHTML = "";
    resetCardState();
  }
  await refreshSessions();
}

async function newSession() {
  const r = await api("/api/sessions", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({}),
  });
  const { id } = await r.json();
  await refreshSessions();
  await loadSession(id);
}

// ---------- chat rendering ----------

function msgNode(role, text, mid, kind) {
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  if (mid != null) div.dataset.mid = mid;
  const who = role === "user"
    ? (kind === "pdf" ? "Врач (подбор pdf)" : "Врач")
    : (kind === "pdf" ? "Подбор pdf" : "Medgemma");
  div.innerHTML = `<div class="who">${who}</div>
    <div class="body"></div>`;
  div.querySelector(".body").innerHTML = fmt(
    role === "user" && kind === "pdf"
      ? `«${pdfPayload(text) || text}»` : text);
  if (role === "user") {
    div.dataset.q = text; // raw question, restored into the composer on rewind
    const btn = document.createElement("button");
    btn.className = "rewind-btn";
    btn.textContent = "↩";
    btn.title = "Вернуться к вопросу";
    btn.onclick = () => rewind(div);
    div.appendChild(btn);
  }
  $("#messages").appendChild(div);
  return div;
}

function noteNode(text, isError, anchor) {
  const div = document.createElement("div");
  div.className = "note" + (isError ? " error" : "");
  div.textContent = text;
  insertAfter(div, anchor || null);
  return div;
}

// sources block: verified guideline links stay visible; everything the
// selection filtered out hides behind ONE collapsible for the whole block
// (labeler 2 is fallible — nothing is lost irrecoverably). A single
// filtered-out entry is shown directly, no toggle.
//
// Pair-verdict revision (owner decision 2026-10-26): every row carries 👎 —
// it NEGATES the pair's current verdict (toggle: attached <-> grayed, the
// pair moves between the normal list and the «отфильтрованные» zone) — and
// 👍 only while the pair was never human-revised (it confirms the machine's
// verdict; after any click only 👎 remains). The CURRENT verdicts map
// (GET /api/verdicts) decides each row's zone and button set, so a
// revision is visible on old turns too; the stored payload is never
// rewritten — history stays as it shipped, the cache changes the future.
const verdictState = { entries: null };

function normDisease(d) {
  return d.trim().toLowerCase().replace(/ё/g, "е"); // server-side key form
}

async function refreshVerdicts() {
  try {
    const r = await fetch("/api/verdicts");
    verdictState.entries = r.ok ? ((await r.json()).entries || {}) : null;
  } catch (e) {
    verdictState.entries = null;
  }
}

function verdictOf(disease, audience, cv) {
  const d = verdictState.entries && verdictState.entries[normDisease(disease)];
  const b = d && d[audience || "none"];
  return (b && b[cv]) || null;
}

function renderAppendix(payload, anchor) {
  const groups = (payload.by_disease || []).filter((g) => (g.picks || []).length);
  const filtered = payload.filtered_out || [];
  // legacy stored payloads carry per-disease no-match notes — dropped
  const notes = (payload.notes || [])
    .filter((n) => !/^Pdf по «.*» не найдено\.$/.test(n));
  if (!(payload.appendix_text || groups.length || filtered.length)) {
    for (const n of notes) noteNode(n, false, anchor || null);
    return null;
  }

  // one flat row model: the stored payload position (picked = right,
  // filtered = wrong) is the DEFAULT; a pair present in the verdicts map
  // renders per its CURRENT verdict instead
  const rows = [];
  const addRow = (c, disease, audience, zone, byline) => {
    const v = verdictOf(disease, audience, c.code_version);
    rows.push({
      code_version: c.code_version, name: c.name, url: c.url,
      disease, audience: audience || "none",
      current: v ? v.verdict : zone,
      human: !!(v && v.human), byline,
    });
  };
  for (const g of groups) {
    const auto = g.auto_pick ? " · автоподбор"
      : g.cached ? " · из кэша" : "";
    const byline = ` — по «${esc(g.disease)}»${auto}`;
    for (const p of g.picks || []) addRow(p, g.disease, g.audience, "right", byline);
  }
  for (const c of filtered) {
    addRow(c, c.disease, c.audience, "wrong", ` — по «${esc(c.disease)}»`);
  }

  const rowHtml = (r, grayed) => `<li${grayed ? ' class="filtered"' : ""}>
    <a href="${esc(r.url)}" target="_blank" rel="noopener">
    «${esc(r.name)}» (${esc(r.code_version)})</a>${r.byline}
    <span class="vd" data-disease="${esc(r.disease)}"
          data-audience="${esc(r.audience)}" data-cv="${esc(r.code_version)}"
          data-current="${r.current}">${
      r.human ? "" : `<button type="button" class="vd-btn" data-act="confirm"
        title="Подтвердить: подобрано верно">👍</button>`
    }<button type="button" class="vd-btn" data-act="negate"
        title="Отклонить подбор (повторное нажатие вернёт)">👎</button></span></li>`;

  const right = rows.filter((r) => r.current === "right");
  const wrong = rows.filter((r) => r.current === "wrong");
  const items = right.map((r) => rowHtml(r, false));
  if (wrong.length === 1) {
    items.push(rowHtml(wrong[0], true));
  } else if (wrong.length > 1) {
    const show = `Показать отфильтрованные (${wrong.length})`;
    items.push(`<li class="cands">
      <button type="button" class="cand-toggle" data-show="${esc(show)}">${esc(show)}</button>
      <ul hidden>${wrong.map((r) => rowHtml(r, true)).join("")}</ul></li>`);
  }

  const wrap = document.createElement("div");
  wrap.className = "sources";
  wrap.innerHTML = `<ul>${items.join("")}</ul>`;
  wrap.addEventListener("click", (e) => {
    const btn = e.target.closest(".cand-toggle");
    if (btn) {
      const ul = btn.parentElement.querySelector("ul");
      ul.hidden = !ul.hidden;
      btn.textContent = ul.hidden ? btn.dataset.show : "Скрыть отфильтрованные";
      return;
    }
    const vd = e.target.closest(".vd-btn");
    if (vd) reviseVerdict(vd.closest(".vd"), vd.dataset.act, wrap, payload, anchor);
  });
  insertAfter(wrap, anchor || null);
  for (const n of notes) noteNode(n, false, anchor || null);
  return wrap;
}

async function reviseVerdict(span, action, wrap, payload, anchor) {
  const { disease, audience, cv, current } = span.dataset;
  // optimistic: apply the same transition the server will persist, then
  // re-render the whole block in place (the row moves zones)
  if (!verdictState.entries) verdictState.entries = {};
  const buckets = verdictState.entries[normDisease(disease)]
    || (verdictState.entries[normDisease(disease)] = {});
  const pairs = buckets[audience] || (buckets[audience] = {});
  const base = (pairs[cv] && pairs[cv].verdict) || current;
  pairs[cv] = {
    verdict: action === "confirm" ? base : (base === "right" ? "wrong" : "right"),
    human: true,
  };
  const fresh = renderAppendix(payload, anchor);
  if (wrap.parentNode && fresh) wrap.parentNode.replaceChild(fresh, wrap);
  try {
    await api("/api/verdicts", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ disease, audience,
        code_version: cv, current, action }),
    });
  } catch (e) {
    noteNode(`Не удалось сохранить оценку: ${e.message}`, true, anchor || null);
    await refreshVerdicts(); // server remains the truth — redraw from it
    const again = renderAppendix(payload, anchor);
    if (fresh && fresh.parentNode && again) {
      fresh.parentNode.replaceChild(again, fresh);
    }
  }
}

// МНН grounding block under the question bubble (owner decision
// 2026-10-13): ONE renderer for both the live `drug_map` SSE event and the
// stored history (user rows carry m.drug_map) — a visible .sources-style
// block that persists across reloads.
function renderDrugMap(matches, anchor) {
  const wrap = document.createElement("div");
  wrap.className = "sources drug";
  // m.en is the rendered per-component МНН (russian + english parens)
  const items = (matches || []).map((m) =>
    `<li>${esc(m.trade)} → ${esc(m.en || m.inn)}</li>`).join("");
  wrap.innerHTML = `<ul>${items}</ul>`;
  insertAfter(wrap, anchor || null);
  return wrap;
}

async function loadSession(id) {
  const run = (async () => {
    state.current = id;
    $("#messages").innerHTML = "";
    resetCardState(); // the visible card state never carries across sessions
    await loadSessionPin(id); // …but the session's pin comes back invisibly
    await refreshVerdicts(); // current pair verdicts for the appendix rows
    const r = await api(`/api/sessions/${id}/messages`);
    const msgs = await r.json();
    // An in-flight turn for THIS session renders from its live wrapper, not
    // from the store: skip the stored rows the wrapper already shows and
    // re-attach it below, so switching away and back keeps the stream on
    // screen (the assistant row lands in the store only when the answer
    // completes — until then the store alone shows just the question).
    const st = activeStream && activeStream.sid === id ? activeStream : null;
    if (st) {
      const last = msgs[msgs.length - 1];
      if (last && last.role === "user" && last.content === st.question) {
        msgs.pop(); // the question row — the wrapper shows it live
      } else if (last && last.role === "assistant") {
        const prev = msgs[msgs.length - 2];
        // pdf turns store the caption row at turn start; chat turns store
        // the answer row at done — either way it follows the question row
        if (prev && prev.role === "user" && prev.content === st.question
            && (st.pdf ? last.kind === "pdf" : st.done)) {
          msgs.length -= 2;
        }
      }
    }
    for (let i = 0; i < msgs.length; i++) {
      const m = msgs[i];
      if (m.role === "user" && m.kind === "pdf") {
        // one row: retitled question bubble, sources block below it; the
        // stored assistant caption row is merged away
        const node = msgNode("user", m.content, m.id, "pdf");
        const next = msgs[i + 1];
        if (next && next.role === "assistant" && next.kind === "pdf") {
          if (next.appendix) renderAppendix(next.appendix, node);
          i++;
        }
        continue;
      }
      const node = msgNode(m.role, m.content, m.id, m.kind);
      if (m.role === "assistant") {
        if (m.reasoning) {
          // restored from the store: collapsed, the physician can expand;
          // rendered with fmt() so markdown emphasis matches the answer
          const t = makeThinkingBlock(node);
          // legacy stored reasoning carries the stray leading marker —
          // " thought\n" (llama-server rows before the 2026-10-25 strip fix)
          // or bare "thought\n" (Ollama era)
          t.body.innerHTML = fmt(m.reasoning.replace(/^ ?thought\n/, ""));
        }
        if (m.appendix) renderAppendix(m.appendix, node);
      } else if (m.drug_map && m.drug_map.length) {
        renderDrugMap(m.drug_map, node); // persisted МНН block under the question
      }
    }
    if (st) $("#messages").appendChild(st.wrap); // resume the live turn in place
    await refreshSessions();
    scrollDown();
  })();
  loadPromise = run;
  return run;
}

// rewind-to-question: delete this question and everything after it,
// put the question back into the composer for editing
async function rewind(node) {
  const mid = node.dataset.mid;
  if (!mid || state.streamActive) return;
  try {
    await api(`/api/sessions/${state.current}/messages/${mid}`, { method: "DELETE" });
  } catch (e) {
    noteNode(`Ошибка удаления: ${e.message}`, true, node);
    return;
  }
  while (node.nextSibling) node.nextSibling.remove();
  node.remove();
  $("#question").value = node.dataset.q;
  $("#question").focus();
  await refreshSessions();
}

function scrollDown() {
  $("#messages").scrollTop = $("#messages").scrollHeight;
}

// Auto-follow the stream only while the view is already at (or near) the
// bottom: scrolling up to read must not be yanked back by the next token.
// The threshold (~80px) covers the couple of lines appended between two
// stream events, so a followed stream stays followed; follow resumes when
// the physician scrolls back down.
function nearBottom() {
  const m = $("#messages");
  return m.scrollHeight - m.scrollTop - m.clientHeight < 80;
}

// ---------- patient card (per-question prompt context; never persisted) ----------

// ONE toggle button (owner decision 2026-10-07): OFF — «Карта пациента»,
// click opens the picker with the pinned (last used) name highlighted;
// ON — shows the patient name, click detaches it (the name stays pinned
// to the page session). While ON every non-pdf question carries the card.
// In-memory only: a reload clears everything; the card travels in the
// chat request, never in the stored text.
const cardState = { card: null, pinned: null }; // card: attached; pinned: last used

function setCard(card) {
  cardState.card = card;
  if (card) {
    cardState.pinned = card;
    // pin immediately on pick (owner decision 2026-10-07): the preselection
    // must survive session switches even when no question was sent; the
    // send path re-pins as a safety net. Best-effort — no session yet means
    // the pin simply lives in memory until the first send.
    if (state.current != null) {
      api(`/api/sessions/${state.current}/card`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ file: card.file }),
      }).catch(() => {});
    }
  }
  const btn = $("#card-btn");
  btn.classList.toggle("card-on", !!card);
  btn.textContent = card ? card.name : "Карта пациента";
  btn.title = card
    ? `Карта пациента: ${card.name} — нажмите, чтобы убрать`
    : (cardState.pinned
        ? `Карта пациента (последняя: ${cardState.pinned.name})`
        : "Карта пациента");
}

// a chat-session switch clears the card entirely (owner decision
// 2026-10-07): both the attached card and the pinned name are per-session
function resetCardState() {
  cardState.pinned = null;
  setCard(null);
  $("#card-menu").hidden = true;
}

// the session's invisible pin (last card used there, stored server-side);
// loadSession fetches it so the picker can preselect it again
async function loadSessionPin(id) {
  try {
    const c = await (await api(`/api/sessions/${id}/card`)).json();
    cardState.pinned = c && c.file ? c : null;
  } catch (e) {
    cardState.pinned = null;
  }
  setCard(cardState.card); // re-render the tooltip with the new session's pin
}

async function toggleCardMenu() {
  const menu = $("#card-menu");
  if (!menu.hidden) { menu.hidden = true; return; }
  let list = null;
  try { list = await (await fetch("/api/cards")).json(); } catch (e) { /* offline */ }
  menu.innerHTML = "";
  if (!list || !list.length) {
    const empty = document.createElement("div");
    empty.className = "card-empty";
    empty.textContent = list ? "Нет сохранённых карт" : "Не удалось загрузить карты";
    menu.appendChild(empty);
  } else {
    // the pinned (last used in this session) card is preselected: moved to
    // the BOTTOM of the list — nearest the click point, the menu opens
    // upward — and highlighted
    const active = cardState.pinned;
    const ordered = active
      ? [...list.filter((c) => c.file !== active.file),
         ...list.filter((c) => c.file === active.file)]
      : list;
    let activeNode = null;
    for (const c of ordered) {
      const item = document.createElement("button");
      item.type = "button";
      const isActive = active && active.file === c.file;
      item.className = "card-item" + (isActive ? " active" : "");
      if (isActive) activeNode = item;
      item.textContent = c.name;
      item.onclick = () => { setCard(c); menu.hidden = true; };
      menu.appendChild(item);
    }
    menu.hidden = false;
    if (activeNode) activeNode.scrollIntoView({ block: "nearest" });
    return;
  }
  menu.hidden = false;
}

// ---------- streaming ----------

function makeThinkingBlock(asst) {
  const details = document.createElement("details");
  details.className = "thinking";
  details.innerHTML = `<summary>Размышления модели</summary><div class="think-body"></div>`;
  asst.insertBefore(details, asst.querySelector(".body"));
  // summary click/keyboard is the only way the user toggles a <details>;
  // programmatic .open changes never fire click, so no self-toggle confusion
  let userToggled = false;
  details.querySelector("summary")
    .addEventListener("click", () => { userToggled = true; }, false);
  return {
    details,
    body: details.querySelector(".think-body"),
    open() {
      if (!userToggled) details.open = true;
    },
    collapseUnlessUserToggled() {
      if (!userToggled) details.open = false;
    },
  };
}

// Elapsed readout under the stop label (owner request 2026-10-17): ticks
// from the send while the turn runs («Останавливаю…» included), then freezes
// and lingers 5 s over «Отправить» fading out. Animation (not transition)
// so a fresh start snaps back to full opacity with nothing to cancel.
let elapsedT0 = 0;
let elapsedInt = null;
let elapsedHideT = null;

function fmtElapsed(ms) {
  const s = Math.floor(ms / 1000);
  const m = Math.floor(s / 60);
  return m ? `${m} мин ${String(s % 60).padStart(2, "0")} с` : `${s} с`;
}

function startElapsed() {
  freezeElapsed();
  const t = $("#send .timer");
  elapsedT0 = Date.now();
  t.hidden = false;
  t.classList.remove("fading");
  t.textContent = "0 с";
  elapsedInt = setInterval(() => {
    t.textContent = fmtElapsed(Date.now() - elapsedT0);
  }, 500);
}

// stop ticking; if the readout is on screen, fade it out over 5 s and hide
function freezeElapsed() {
  if (elapsedInt) { clearInterval(elapsedInt); elapsedInt = null; }
  if (elapsedHideT) { clearTimeout(elapsedHideT); elapsedHideT = null; }
  const t = $("#send .timer");
  if (t.hidden || t.classList.contains("fading")) return;
  if (elapsedT0) t.textContent = fmtElapsed(Date.now() - elapsedT0);
  t.classList.add("fading");
  elapsedHideT = setTimeout(() => {
    t.hidden = true;
    t.classList.remove("fading");
  }, 5000);
}

function setSendMode(mode) { // "send" | "stop" | "stopping"
  const btn = $("#send");
  btn.classList.toggle("stop", mode !== "send");
  btn.disabled = mode === "stopping";
  btn.querySelector(".label").textContent = mode === "send" ? "Отправить"
    : mode === "stop" ? "Стоп" : "Останавливаю…";
  if (mode === "send") freezeElapsed();
}

async function stopTurn() {
  if (!state.busy || !activeStream || activeStream.done || stopping) return;
  stopping = true;
  setSendMode("stopping");
  try {
    await api(`/api/sessions/${activeStream.sid}/stop`, { method: "POST" });
    // the stream itself ends with a `stopped` event (or an honest failure)
  } catch (e) {
    stopping = false; // the stream keeps running — let the physician retry
    setSendMode("stop");
    noteNode(`Не удалось остановить: ${e.message}`, true,
             activeStream.wrap || null);
  }
}

async function send(question) {
  await loadPromise; // never race a session history load (it would wipe our nodes)
  state.busy = true;
  state.streamActive = true; // covers the whole SSE stream, appendix included
  const seq = ++sendSeq;
  stopping = false;
  let gotStopped = false;
  setSendMode("stop");
  startElapsed();
  // The whole live turn — question bubble, thinking, answer, status lines,
  // candidate cards, sources — lives in ONE wrapper. A session switch wipes
  // #messages and detaches the wrapper whole; coming back re-attaches it and
  // the stream stays visible instead of silently writing into orphaned nodes
  // (and stage lines leaking into the wrong session's view via insertAfter).
  const wrap = document.createElement("div");
  const userNode = msgNode("user", question);
  const asst = msgNode("assistant", "");
  wrap.append(userNode, asst);
  $("#messages").appendChild(wrap);
  activeStream = { sid: state.current, wrap, question, pdf: false, done: false };
  const body = asst.querySelector(".body");
  body.style.display = "none"; // the answer block appears only when the answer starts
  let thinking = null;
  let thinkRaw = ""; // raw reasoning text; rendered through fmt() like the answer
  let statusLine = null;
  let answer = "";
  let anchor = asst; // pdf turns retarget status/sources/error nodes to the user bubble
  // only scroll the chat while our turn is actually on screen — a background
  // stream in another session must not yank the physician's current view —
  // and only while the view is at the bottom (nearBottom): scrolling up
  // pauses the follow until the physician returns to the bottom
  const autoScroll = () => { if (wrap.isConnected && nearBottom()) scrollDown(); };
  // Follow around a DOM mutation (owner rule 2026-10-26): the follow decision
  // is the PRE-mutation position, not the post-mutation one. A stage event
  // can add more than the nearBottom threshold in one block — the candidates
  // list is 120-160px — and checking after the insert broke the follow
  // exactly there: the «Кандидаты pdf…» block and the status line below it
  // stayed under the fold for the rest of the turn while the answer stream
  // had followed fine (token deltas fit inside the threshold). Manual
  // scrolling up still pauses the follow; it resumes at the bottom.
  const followScroll = (mutate) => {
    const on = wrap.isConnected && nearBottom();
    mutate();
    if (on) scrollDown();
  };
  scrollDown();

  let sid = state.current;
  try {
    if (sid == null) {
      const r = await api("/api/sessions", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title: question.slice(0, 60) }),
      });
      sid = (await r.json()).id;
      state.current = sid;
    }
    activeStream.sid = sid;
    // the card applies while a patient is selected and the message is not
    // a pdf turn (belt and suspenders — the server also ignores that path)
    const useCard = cardState.card && pdfPayload(question) === "";
    const resp = await fetch("/api/chat", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(useCard
        ? { session_id: sid, question, card: cardState.card.file }
        : { session_id: sid, question }),
    });
    if (!resp.ok || !resp.body) throw new Error(`${resp.status} ${await resp.text()}`);

    const reader = resp.body.getReader();
    const dec = new TextDecoder();
    let buf = "", cursorOn = true, answerStarted = false;
    const render = () => followScroll(() => {
      body.innerHTML = fmt(answer) + (cursorOn ? `<span class="cursor">▍</span>` : "");
    });
    const status = (text) => followScroll(() => {
      if (!statusLine) {
        statusLine = document.createElement("div");
        statusLine.className = "status-line";
        insertAfter(statusLine, anchor);
      }
      statusLine.textContent = text;
    });
    const clearStatus = () => {
      if (statusLine) { statusLine.remove(); statusLine = null; }
    };

    // candidate cards arrive with the select stage (search just finished);
    // the final appendix event replaces them with the verified picks
    let candBlocks = [];
    let candAnchor = asst;
    const showCandidates = (disease, cards) => {
      followScroll(() => {
        const candWrap = document.createElement("div");
        candWrap.className = "sources provisional";
        // Decided pairs arrive in their decided state (owner catch
        // 2026-10-26 evening): a cached WRONG verdict — the doctor's 👎 or
        // an earlier filter — grays out right here for the whole select
        // cycle instead of pretending to be reconsidered; a cached RIGHT
        // pair is already attaching; only the unmarked rows are actually
        // being filtered
        const items = (cards || []).map((c) => {
          const decided = c.verdict === "wrong" ? " filtered"
            : c.verdict === "right" ? " cand-ok" : "";
          const mark = c.verdict === "wrong" ? " · отклонено"
            : c.verdict === "right" ? " · уже отобрано" : "";
          return `<li class="cand${decided}">${
            c.url
              ? `<a href="${esc(c.url)}" target="_blank" rel="noopener">«${esc(c.name)}» (${esc(c.code_version)})</a>`
              : `«${esc(c.name)}» (${esc(c.code_version)})`
          }<span class="cand-note">${mark}</span></li>`;
        }).join("");
        candWrap.innerHTML = `<div class="caption">Кандидаты pdf — фильтрую по «${esc(disease)}»</div>
          <ul>${items}</ul>`;
        insertAfter(candWrap, candAnchor);
        candAnchor = candWrap;
        candBlocks.push(candWrap);
      });
    };
    const clearCandidates = () => {
      candBlocks.forEach((b) => b.remove());
      candBlocks = [];
      candAnchor = anchor;
    };

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let idx;
      while ((idx = buf.indexOf("\n\n")) !== -1) {
        const block = buf.slice(0, idx);
        buf = buf.slice(idx + 2);
        const ev = { event: "message", data: "{}" };
        for (const line of block.split("\n")) {
          if (line.startsWith("event: ")) ev.event = line.slice(7).trim();
          else if (line.startsWith("data: ")) ev.data = line.slice(6);
        }
        const data = JSON.parse(ev.data);
        if (ev.event === "reasoning") {
          if (!thinking) thinking = makeThinkingBlock(asst);
          followScroll(() => {
            thinkRaw += data.t;
            thinking.body.innerHTML = fmt(thinkRaw);
            thinking.open();
          });
        } else if (ev.event === "reasoning_reset") {
          // degenerate thinking loop aborted upstream; the retry restarts
          // the reasoning from scratch (or the turn died on a second loop) —
          // drop the looped garbage, leave a one-line trace of the restart
          if (!thinking) thinking = makeThinkingBlock(asst);
          thinkRaw = data.note ? data.note + "\n\n" : "";
          thinking.body.innerHTML = fmt(thinkRaw);
        } else if (ev.event === "drug_map") {
          // register-grounded МНН block under the question bubble; the
          // same renderer restores it from m.drug_map after a reload
          followScroll(() => renderDrugMap(data.matches, userNode));
        } else if (ev.event === "kind" && data.turn === "pdf") {
          // manual pdf turn renders as ONE row: the question bubble is
          // retitled and re-bodied; the assistant bubble never appears
          activeStream.pdf = true;
          const who = userNode.querySelector(".who");
          if (who) who.textContent = "Врач (подбор pdf)";
          userNode.querySelector(".body").innerHTML = fmt(`«${data.payload}»`);
          asst.remove();
          anchor = userNode;
          candAnchor = userNode;
        } else if (ev.event === "token") {
          if (!answerStarted) {
            answerStarted = true;
            body.style.display = "";
            body.innerHTML = `<span class="cursor">▍</span>`;
            if (thinking) thinking.collapseUnlessUserToggled();
          }
          answer += data.t;
          render();
        } else if (ev.event === "stage") {
          status(data.text);
          if (data.stage === "select" && data.cards) showCandidates(data.disease, data.cards);
        } else if (ev.event === "done") {
          // answer complete: the composer unblocks while the appendix finishes
          state.busy = false;
          setSendMode("send");
          cursorOn = false;
          render();
          // ids are set even while detached: they must survive a session
          // switch so rewind keeps working on the re-attached turn
          if (data.user_message_id != null) userNode.dataset.mid = data.user_message_id;
          if (data.message_id != null) asst.dataset.mid = data.message_id;
          activeStream.done = true;
        } else if (ev.event === "appendix") {
          followScroll(() => {
            clearStatus();
            clearCandidates(); // provisional list -> final filtered sources
            renderAppendix(data, anchor);
          });
        } else if (ev.event === "stopped") {
          // undo the send (owner decision 2026-10-15): the server deleted the
          // turn's rows; drop the live wrapper and put the question back into
          // the composer — but only when this turn is the one on screen (a
          // background stream in another session must not clobber the input)
          gotStopped = true;
          stopping = false;
          const onScreen = wrap.isConnected && state.current === sid;
          wrap.remove();
          if (onScreen) {
            $("#question").value = question;
            $("#question").focus();
          }
        } else if (ev.event === "error") {
          followScroll(() => {
            clearStatus();
            clearCandidates();
            cursorOn = false;
            render();
            noteNode(data.text, true, anchor);
          });
        }
      }
    }
    cursorOn = false;
    render();
    clearStatus();
    clearCandidates(); // stream ended without a final appendix payload
  } catch (e) {
    clearCandidates();
    body.innerHTML = fmt(answer);
    noteNode(`Ошибка запроса: ${e.message}`, true, anchor);
  } finally {
    const unconfirmed = stopping && !gotStopped; // stop sent, no `stopped` event
    if (seq === sendSeq) {
      // only the LATEST turn may govern the composer — an earlier turn's
      // stream can still be finishing its appendix after a newer send began
      state.busy = false;
      state.streamActive = false;
      setSendMode("send");
      stopping = false;
      activeStream = null; // the turn is fully stored now — normal render takes over
      if (unconfirmed) {
        noteNode("Остановка не подтвердилась — ход сохранён; " +
                 "можно вернуться к вопросу ↩.", false, null);
      }
    }
    await refreshSessions();
    autoScroll();
  }
}

// ---------- boot ----------

// A stale cached index.html must not wedge the composer (session 84,
// 2026-10-17): the server's no-cache revalidation covered .js/.css but
// not .html, so a reload paired fresh app.js with the old flat button —
// setSendMode's .label lookup threw before send()'s try block and busy
// stuck on (question eaten, no nodes, no fetch). Rebuild the structure
// here so the spans always exist, whatever markup the browser cached.
(function ensureSendParts() {
  const btn = $("#send");
  if (btn.querySelector(".label")) return;
  const label = document.createElement("span");
  label.className = "label";
  label.textContent = btn.textContent.trim();
  const timer = document.createElement("span");
  timer.className = "timer";
  timer.hidden = true;
  btn.textContent = "";
  btn.append(label, timer);
})();

$("#composer").addEventListener("submit", (e) => {
  e.preventDefault();
  if (state.busy) { stopTurn(); return; } // the send button is «Стоп» while busy
  const q = $("#question").value.trim();
  if (!q) return;
  $("#question").value = "";
  send(q);
});
$("#question").addEventListener("keydown", (e) => {
  if (e.key === "Escape" && state.busy) {
    e.preventDefault();
    stopTurn();
    return;
  }
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    $("#composer").requestSubmit();
  }
});
$("#new-session").addEventListener("click", newSession);
$("#card-btn").addEventListener("click", () => {
  if (cardState.card) setCard(null); // ON click detaches; the name stays pinned
  else toggleCardMenu();             // OFF click opens the picker
});
document.addEventListener("click", (e) => {
  if (!e.target.closest("#card-btn") && !e.target.closest("#card-menu"))
    $("#card-menu").hidden = true;
});

refreshSessions().then(() => { if (state.sessions.length) loadSession(state.sessions[0].id); });
