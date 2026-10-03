"use strict";

// Shared client utilities for the chat page and the admin page. Escaping
// lives in ONE place: an XSS fix or null-handling change applies everywhere.
function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}
