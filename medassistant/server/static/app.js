"use strict";

const $ = (sel) => document.querySelector(sel);
const state = { sessions: [], current: null, busy: false, streamActive: false };
let loadPromise = Promise.resolve();

// ---------- utilities ----------

function esc(s) {
  return s.replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

function fmt(text) {
  return esc(text)
    .replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>")
    .replace(/\*([^*\n]+)\*/g, "<i>$1</i>");
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

function msgNode(role, text, mid) {
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  if (mid != null) div.dataset.mid = mid;
  div.innerHTML = `<div class="who">${role === "user" ? "Врач" : "Medgemma"}</div>
    <div class="body"></div>`;
  div.querySelector(".body").innerHTML = fmt(text);
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
      const byDisease = ` — по заболеванию «${esc(g.disease)}»${auto}`;
      for (const p of g.picks || []) {
        items.push(`<li><a href="${esc(p.url)}" target="_blank" rel="noopener">
          «${esc(p.name)}» (${esc(p.code_version)})</a>${byDisease}</li>`);
      }
    }
    const filteredItem = (c) => `<li class="filtered">
      <a href="${esc(c.url)}" target="_blank" rel="noopener">
      «${esc(c.name)}» (${esc(c.code_version)})</a> — по заболеванию «${esc(c.disease)}»</li>`;
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

async function loadSession(id) {
  const run = (async () => {
    state.current = id;
    $("#messages").innerHTML = "";
    const r = await api(`/api/sessions/${id}/messages`);
    for (const m of await r.json()) {
      const node = msgNode(m.role, m.content, m.id);
      if (m.role === "assistant") {
        if (m.reasoning) {
          // restored from the store: collapsed, the physician can expand
          const t = makeThinkingBlock(node);
          // legacy stored reasoning carries the stray leading "thought\n"
          t.body.textContent = m.reasoning.replace(/^thought\n/, "");
        }
        if (m.appendix) renderAppendix(m.appendix, node);
      }
    }
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
  const userNode = msgNode("user", question);
  const asst = msgNode("assistant", "");
  const body = asst.querySelector(".body");
  body.style.display = "none"; // the answer block appears only when the answer starts
  let thinking = null;
  let statusLine = null;
  let answer = "";
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
    const resp = await fetch("/api/chat", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sid, question }),
    });
    if (!resp.ok || !resp.body) throw new Error(`${resp.status} ${await resp.text()}`);

    const reader = resp.body.getReader();
    const dec = new TextDecoder();
    let buf = "", cursorOn = true, answerStarted = false;
    const render = () => {
      body.innerHTML = fmt(answer) + (cursorOn ? `<span class="cursor">▍</span>` : "");
      scrollDown();
    };
    const status = (text) => {
      if (!statusLine) {
        statusLine = document.createElement("div");
        statusLine.className = "status-line";
        insertAfter(statusLine, asst);
      }
      statusLine.textContent = text;
      scrollDown();
    };
    const clearStatus = () => {
      if (statusLine) { statusLine.remove(); statusLine = null; }
    };

    // candidate cards arrive with the select stage (search just finished);
    // the final appendix event replaces them with the verified picks
    let candBlocks = [];
    let candAnchor = asst;
    const showCandidates = (disease, cards) => {
      const wrap = document.createElement("div");
      wrap.className = "sources provisional";
      const items = (cards || []).map((c) => `<li>${
        c.url
          ? `<a href="${esc(c.url)}" target="_blank" rel="noopener">«${esc(c.name)}» (${esc(c.code_version)})</a>`
          : `«${esc(c.name)}» (${esc(c.code_version)})`
      }</li>`).join("");
      wrap.innerHTML = `<div class="caption">Кандидаты pdf — фильтрую по заболеванию «${esc(disease)}»</div>
        <ul>${items}</ul>`;
      insertAfter(wrap, candAnchor);
      candAnchor = wrap;
      candBlocks.push(wrap);
      scrollDown();
    };
    const clearCandidates = () => {
      candBlocks.forEach((b) => b.remove());
      candBlocks = [];
      candAnchor = asst;
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
          scrollDown();
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
          if (data.user_message_id != null) userNode.dataset.mid = data.user_message_id;
          if (data.message_id != null) asst.dataset.mid = data.message_id;
        } else if (ev.event === "appendix") {
          clearStatus();
          clearCandidates(); // provisional list -> final filtered sources
          renderAppendix(data, asst);
          scrollDown();
        } else if (ev.event === "error") {
          clearStatus();
          clearCandidates();
          cursorOn = false;
          render();
          noteNode(data.text, true, asst);
          scrollDown();
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
    noteNode(`Ошибка запроса: ${e.message}`, true, asst);
  } finally {
    state.busy = false;
    state.streamActive = false;
    $("#send").disabled = false;
    await refreshSessions();
    scrollDown();
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

refreshSessions().then(() => { if (state.sessions.length) loadSession(state.sessions[0].id); });
