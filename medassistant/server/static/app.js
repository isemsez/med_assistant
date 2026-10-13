"use strict";

const $ = (sel) => document.querySelector(sel);
const state = { sessions: [], current: null, busy: false, streamActive: false };
let loadPromise = Promise.resolve();
// The in-flight turn: {sid, wrap, question, pdf, done}. loadSession wipes
// #messages on every switch, so the live nodes must be re-attachable —
// while it streams, the store holds at most the question (the assistant
// row with its reasoning lands only when the answer completes).
let activeStream = null;

// esc() lives in util.js, loaded before this script.

function fmt(text) {
  return esc(text)
    .replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>")
    .replace(/\*([^*\n]+)\*/g, "<i>$1</i>");
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
function renderAppendix(payload, anchor) {
  const groups = (payload.by_disease || []).filter((g) => (g.picks || []).length);
  const filtered = payload.filtered_out || [];
  if (payload.appendix_text || groups.length || filtered.length) {
    const wrap = document.createElement("div");
    wrap.className = "sources";
    const items = [];
    for (const g of groups) {
      const auto = g.auto_pick ? " · автоподбор" : "";
      const byDisease = ` — по «${esc(g.disease)}»${auto}`;
      for (const p of g.picks || []) {
        items.push(`<li><a href="${esc(p.url)}" target="_blank" rel="noopener">
          «${esc(p.name)}» (${esc(p.code_version)})</a>${byDisease}</li>`);
      }
    }
    const filteredItem = (c) => `<li class="filtered">
      <a href="${esc(c.url)}" target="_blank" rel="noopener">
      «${esc(c.name)}» (${esc(c.code_version)})</a> — по «${esc(c.disease)}»</li>`;
    if (filtered.length === 1) {
      items.push(filteredItem(filtered[0]));
    } else if (filtered.length > 1) {
      const show = `Показать отфильтрованные (${filtered.length})`;
      items.push(`<li class="cands">
        <button type="button" class="cand-toggle" data-show="${esc(show)}">${esc(show)}</button>
        <ul hidden>${filtered.map(filteredItem).join("")}</ul></li>`);
    }
    wrap.addEventListener("click", (e) => {
      const btn = e.target.closest(".cand-toggle");
      if (!btn) return;
      const ul = btn.parentElement.querySelector("ul");
      ul.hidden = !ul.hidden;
      btn.textContent = ul.hidden ? btn.dataset.show : "Скрыть отфильтрованные";
    });
    wrap.innerHTML = `<ul>${items.join("")}</ul>`;
    insertAfter(wrap, anchor || null);
  }
  for (const n of payload.notes || []) {
    // legacy stored payloads carry per-disease no-match notes — dropped
    if (/^Pdf по «.*» не найдено\.$/.test(n)) continue;
    noteNode(n, false, anchor || null);
  }
}

// МНН grounding block under the question bubble (owner decision
// 2026-10-13): ONE renderer for both the live `drug_map` SSE event and the
// stored history (user rows carry m.drug_map) — a visible .sources-style
// block that persists across reloads.
function renderDrugMap(matches, anchor) {
  const wrap = document.createElement("div");
  wrap.className = "sources drug";
  const items = (matches || []).map((m) =>
    `<li>${esc(m.trade)} → ${esc(m.inn)}</li>`).join("");
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
          // restored from the store: collapsed, the physician can expand
          const t = makeThinkingBlock(node);
          // legacy stored reasoning carries the stray leading "thought\n"
          t.body.textContent = m.reasoning.replace(/^thought\n/, "");
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

async function send(question) {
  await loadPromise; // never race a session history load (it would wipe our nodes)
  state.busy = true;
  state.streamActive = true; // covers the whole SSE stream, appendix included
  $("#send").disabled = true;
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
  let statusLine = null;
  let answer = "";
  let anchor = asst; // pdf turns retarget status/sources/error nodes to the user bubble
  // only scroll the chat while our turn is actually on screen — a background
  // stream in another session must not yank the physician's current view —
  // and only while the view is at the bottom (nearBottom): scrolling up
  // pauses the follow until the physician returns to the bottom
  const autoScroll = () => { if (wrap.isConnected && nearBottom()) scrollDown(); };
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
    const render = () => {
      body.innerHTML = fmt(answer) + (cursorOn ? `<span class="cursor">▍</span>` : "");
      autoScroll();
    };
    const status = (text) => {
      if (!statusLine) {
        statusLine = document.createElement("div");
        statusLine.className = "status-line";
        insertAfter(statusLine, anchor);
      }
      statusLine.textContent = text;
      autoScroll();
    };
    const clearStatus = () => {
      if (statusLine) { statusLine.remove(); statusLine = null; }
    };

    // candidate cards arrive with the select stage (search just finished);
    // the final appendix event replaces them with the verified picks
    let candBlocks = [];
    let candAnchor = asst;
    const showCandidates = (disease, cards) => {
      const candWrap = document.createElement("div");
      candWrap.className = "sources provisional";
      const items = (cards || []).map((c) => `<li>${
        c.url
          ? `<a href="${esc(c.url)}" target="_blank" rel="noopener">«${esc(c.name)}» (${esc(c.code_version)})</a>`
          : `«${esc(c.name)}» (${esc(c.code_version)})`
      }</li>`).join("");
      candWrap.innerHTML = `<div class="caption">Кандидаты pdf — фильтрую по «${esc(disease)}»</div>
        <ul>${items}</ul>`;
      insertAfter(candWrap, candAnchor);
      candAnchor = candWrap;
      candBlocks.push(candWrap);
      autoScroll();
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
          thinking.body.textContent += data.t;
          thinking.open();
          autoScroll();
        } else if (ev.event === "reasoning_reset") {
          // degenerate thinking loop aborted upstream; the retry restarts
          // the reasoning from scratch — drop the looped garbage
          if (thinking) thinking.body.textContent = "";
        } else if (ev.event === "drug_map") {
          // register-grounded МНН block under the question bubble; the
          // same renderer restores it from m.drug_map after a reload
          renderDrugMap(data.matches, userNode);
          autoScroll();
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
          $("#send").disabled = false;
          cursorOn = false;
          render();
          // ids are set even while detached: they must survive a session
          // switch so rewind keeps working on the re-attached turn
          if (data.user_message_id != null) userNode.dataset.mid = data.user_message_id;
          if (data.message_id != null) asst.dataset.mid = data.message_id;
          activeStream.done = true;
        } else if (ev.event === "appendix") {
          clearStatus();
          clearCandidates(); // provisional list -> final filtered sources
          renderAppendix(data, anchor);
          autoScroll();
        } else if (ev.event === "error") {
          clearStatus();
          clearCandidates();
          cursorOn = false;
          render();
          noteNode(data.text, true, anchor);
          autoScroll();
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
    state.busy = false;
    state.streamActive = false;
    $("#send").disabled = false;
    activeStream = null; // the turn is fully stored now — normal render takes over
    await refreshSessions();
    autoScroll();
  }
}

// ---------- boot ----------

$("#composer").addEventListener("submit", (e) => {
  e.preventDefault();
  const q = $("#question").value.trim();
  if (!q || state.busy) return;
  $("#question").value = "";
  send(q);
});
$("#question").addEventListener("keydown", (e) => {
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
