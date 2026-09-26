(() => {
  const TOKEN_KEY = "sh_token";
  const USER_KEY = "sh_user";
  let offset = 0;
  const pageSize = 25;
  let currentPatient = null;

  const $ = (id) => document.getElementById(id);
  const loginEl = $("login");
  const appEl = $("app");

  function token() {
    return localStorage.getItem(TOKEN_KEY);
  }

  async function api(path, opts = {}) {
    const headers = Object.assign({ Accept: "application/json" }, opts.headers || {});
    const t = token();
    if (t) headers.Authorization = `Bearer ${t}`;
    if (opts.body && !(opts.body instanceof URLSearchParams)) {
      headers["Content-Type"] = "application/json";
    }
    const res = await fetch(path, { ...opts, headers });
    if (res.status === 401) {
      logout(false);
      throw new Error("Session expired — sign in again");
    }
    const text = await res.text();
    let data = null;
    try { data = text ? JSON.parse(text) : null; } catch { data = text; }
    if (!res.ok) {
      const detail = data && data.detail ? JSON.stringify(data.detail) : res.statusText;
      throw new Error(detail || `HTTP ${res.status}`);
    }
    return data;
  }

  function showApp() {
    loginEl.hidden = true;
    appEl.hidden = false;
    const user = JSON.parse(localStorage.getItem(USER_KEY) || "{}");
    $("who").textContent = `${user.username || "user"} · ${user.role || ""}`;
  }

  function logout(clearForm = true) {
    localStorage.removeItem(TOKEN_KEY);
    localStorage.removeItem(USER_KEY);
    appEl.hidden = true;
    loginEl.hidden = false;
    if (clearForm) $("login-err").hidden = true;
  }

  $("login-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const fd = new FormData(e.target);
    const err = $("login-err");
    err.hidden = true;
    try {
      const tok = await api("/auth/token", {
        method: "POST",
        body: JSON.stringify({
          username: fd.get("username"),
          password: fd.get("password"),
        }),
      });
      localStorage.setItem(TOKEN_KEY, tok.access_token);
      localStorage.setItem(USER_KEY, JSON.stringify({ username: fd.get("username"), role: tok.role }));
      showApp();
      await loadPatients();
    } catch (ex) {
      err.textContent = ex.message;
      err.hidden = false;
    }
  });

  $("logout").addEventListener("click", () => logout());

  async function loadPatients(query) {
    const list = $("patient-list");
    list.innerHTML = "<li class='muted'>Loading…</li>";
    let url = `/fhir/Patient?_count=${pageSize}&_offset=${offset}`;
    if (query) {
      const q = String(query).trim();
      if (/^\d+$/.test(q)) url = `/fhir/Patient/${q}`;
      else url = `/fhir/Patient?name=${encodeURIComponent(q)}&_count=${pageSize}`;
    }
    try {
      const data = await api(url);
      list.innerHTML = "";
      const entries = data.resourceType === "Patient"
        ? [{ resource: data }]
        : (data.entry || []);
      if (!entries.length) {
        list.innerHTML = "<li class='muted'>No patients found.</li>";
        return;
      }
      for (const e of entries) {
        const p = e.resource;
        const li = document.createElement("li");
        const btn = document.createElement("button");
        btn.type = "button";
        const name = (p.name && p.name[0] && p.name[0].text) || `Patient ${p.id}`;
        btn.innerHTML = `<strong>${esc(name)}</strong><span class="meta">MRN ${esc(p.id)} · ${esc(p.gender || "")} · DOB ${esc(p.birthDate || "")}</span>`;
        btn.addEventListener("click", () => {
          list.querySelectorAll("button").forEach((b) => b.classList.remove("active"));
          btn.classList.add("active");
          openChart(p.id);
        });
        li.appendChild(btn);
        list.appendChild(li);
      }
    } catch (ex) {
      list.innerHTML = `<li class="err">${esc(ex.message)}</li>`;
    }
  }

  $("btn-search").addEventListener("click", () => {
    offset = 0;
    loadPatients($("q").value);
  });
  $("q").addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      offset = 0;
      loadPatients($("q").value);
    }
  });
  $("prev").addEventListener("click", () => {
    offset = Math.max(0, offset - pageSize);
    loadPatients($("q").value);
  });
  $("next").addEventListener("click", () => {
    offset += pageSize;
    loadPatients($("q").value);
  });

  async function openChart(patientId) {
    currentPatient = patientId;
    const root = $("chart");
    root.innerHTML = "<p class='muted'>Opening chart…</p>";
    try {
      const [summary, encounters] = await Promise.all([
        api(`/epic/chart/${patientId}/summary`),
        api(`/epic/chart/${patientId}/encounters`),
      ]);
      root.innerHTML = `
        <div class="chart-head">
          <div>
            <h1>${esc(summary.name || "Patient " + patientId)}</h1>
            <div class="demog">
              ${esc(summary.age)}y ${esc(summary.sex)} · ${esc(summary.race_ethnicity || "")}
              · ${esc(summary.insurance || "")}
            </div>
            <div style="margin-top:0.5rem">
              <span class="badge">MRN ${esc(String(patientId))}</span>
              ${summary.pcp ? `<span class="badge">PCP ${esc(summary.pcp)}</span>` : ""}
            </div>
          </div>
        </div>
        <div class="grid-2">
          <div class="card">
            <h3>Problem list</h3>
            <ul class="problems" id="problems"></ul>
          </div>
          <div class="card">
            <h3>Encounters</h3>
            <div id="encounters"></div>
          </div>
        </div>
        <div class="sections" id="sections">
          <p class="muted">Select an encounter to read the note.</p>
        </div>
      `;
      const probs = $("problems");
      for (const p of summary.active_problems || []) {
        const li = document.createElement("li");
        li.innerHTML = `<strong>${esc(p.display_name)}</strong> <span class="meta">${esc(p.icd10_code || "")}</span>`;
        probs.appendChild(li);
      }
      if (!(summary.active_problems || []).length) probs.innerHTML = "<li class='muted'>None</li>";

      const encRoot = $("encounters");
      const sorted = [...(encounters || [])].sort((a, b) => String(b.date).localeCompare(String(a.date)));
      for (const enc of sorted) {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "enc";
        btn.innerHTML = `<div class="when">${esc(enc.date)} · ${esc(enc.type)} · ${esc(enc.department || "")}</div>
          <div class="cc">${esc(enc.chief_complaint || "")}</div>
          <div class="meta">Attending: ${esc(enc.attending || "—")}</div>`;
        btn.addEventListener("click", () => {
          encRoot.querySelectorAll(".enc").forEach((b) => b.classList.remove("active"));
          btn.classList.add("active");
          openEncounter(patientId, enc.encounter_id);
        });
        encRoot.appendChild(btn);
      }
    } catch (ex) {
      root.innerHTML = `<p class="err">${esc(ex.message)}</p>`;
    }
  }

  async function openEncounter(patientId, encounterId) {
    const root = $("sections");
    root.innerHTML = "<p class='muted'>Loading note…</p>";
    try {
      const enc = await api(`/epic/chart/${patientId}/encounters/${encounterId}`);
      const sections = [...(enc.sections || [])].sort((a, b) => (a.section_order || 0) - (b.section_order || 0));
      root.innerHTML = `<h3 style="margin:0 0 0.5rem">${esc(enc.date)} — ${esc(enc.chief_complaint || "Encounter")}</h3>`;
      for (const s of sections) {
        const div = document.createElement("div");
        div.className = "section";
        div.innerHTML = `<h4>${esc(prettySection(s.section_type))}</h4><pre>${esc(s.section_text || "")}</pre>`;
        root.appendChild(div);
      }
      if (!sections.length) root.innerHTML += "<p class='muted'>No sections visible for this role.</p>";
    } catch (ex) {
      root.innerHTML = `<p class="err">${esc(ex.message)}</p>`;
    }
  }

  function prettySection(t) {
    return String(t || "").replace(/_/g, " ");
  }

  function esc(s) {
    return String(s == null ? "" : s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  if (token()) {
    showApp();
    loadPatients();
  }
})();
