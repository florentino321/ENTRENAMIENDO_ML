/**
 * app.js — Lógica del Dashboard PaveDetect AI
 * Maneja navegación, llamadas a la API Flask y visualizaciones con Chart.js
 */

const API = "";  // Misma URL base (Flask sirve el frontend)

// ─────────────────────────────────────────────────────────
// Utilidades globales
// ─────────────────────────────────────────────────────────
function toast(msg, tipo = "info") {
  const icons = { success: "bi-check-circle-fill", error: "bi-x-circle-fill", info: "bi-info-circle-fill", warn: "bi-exclamation-triangle-fill" };
  const el = document.createElement("div");
  el.className = `toast toast-${tipo}`;
  el.innerHTML = `<i class="bi ${icons[tipo]}"></i><span class="toast-msg">${msg}</span>`;
  document.getElementById("toastContainer").appendChild(el);
  setTimeout(() => el.remove(), 4000);
}

function fmt(val, suf = "") { return val !== undefined && val !== null ? `${val}${suf}` : "—"; }
function fmtFecha(s) { if (!s) return "—"; const d = new Date(s + (s.includes("T") ? "" : "Z")); return d.toLocaleDateString("es-PE", { day: "2-digit", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit" }); }

// ─────────────────────────────────────────────────────────
// Navegación por pestañas
// ─────────────────────────────────────────────────────────
const tabTitles = { dashboard: "Dashboard", inferencia: "Analizar Imagen", entrenamiento: "Entrenar CNN", regresion: "Regresión Lineal", historial: "Historial", sesiones: "Sesiones" };
let currentTab = "dashboard";

function switchTab(tabId) {
  document.querySelectorAll(".tab-content").forEach(t => t.classList.remove("active"));
  document.querySelectorAll(".nav-item").forEach(n => n.classList.remove("active"));
  document.getElementById(`tab-${tabId}`).classList.add("active");
  document.getElementById(`nav-${tabId}`)?.classList.add("active");
  document.getElementById("topbarTitle").textContent = tabTitles[tabId] || tabId;
  currentTab = tabId;
  // Cargar datos al cambiar de pestaña
  if (tabId === "dashboard")    { cargarEstadisticas(); cargarHistorial(); }
  if (tabId === "historial")    cargarTablaHistorial();
  if (tabId === "sesiones")     cargarSesiones();
  if (tabId === "regresion")    cargarDatosRegresion();
}

document.querySelectorAll(".nav-item").forEach(item => {
  item.addEventListener("click", e => { e.preventDefault(); switchTab(item.dataset.tab); });
});

// Toggle sidebar
const sidebar = document.getElementById("sidebar");
const mainContent = document.getElementById("mainContent");
document.getElementById("sidebarToggle").addEventListener("click", () => {
  sidebar.classList.toggle("collapsed");
  mainContent.classList.toggle("expanded");
});

// ─────────────────────────────────────────────────────────
// Instancias Chart.js
// ─────────────────────────────────────────────────────────
let chartCat = null, chartAct = null, chartTrain = null, chartReg = null, chartDoc = null;

const CHART_DEFAULTS = {
  color: "#94a3b8",
  plugins: { legend: { labels: { color: "#94a3b8", font: { family: "Inter" } } } },
};

// ─────────────────────────────────────────────────────────
// DASHBOARD — Estadísticas y KPIs
// ─────────────────────────────────────────────────────────
async function cargarEstadisticas() {
  try {
    const res  = await fetch(`${API}/api/estadisticas`);
    const data = await res.json();
    if (!data.ok) return;
    const d = data.datos;

    document.getElementById("kpiTotal").textContent          = d.total_analizadas || 0;
    document.getElementById("kpiConfianza").textContent      = `${d.promedio_confianza || 0}%`;
    document.getElementById("kpiEntrenamientos").textContent = d.total_entrenamientos || 0;
    document.getElementById("kpiModelo").textContent         = data.modelo_listo ? "Listo ✓" : "Sin entrenar";

    // Gráfico donut de categorías
    const cats   = d.por_categoria || [];
    const labels = cats.map(c => c.nombre || c[0]);
    const vals   = cats.map(c => c.cantidad || c[1]);
    const colors = cats.map(c => c.color_hex || c[2] || "#6366f1");

    if (chartCat) chartCat.destroy();
    chartCat = new Chart(document.getElementById("chartCategorias"), {
      type: "doughnut",
      data: { labels, datasets: [{ data: vals, backgroundColor: colors, borderColor: "#1a2234", borderWidth: 3, hoverOffset: 8 }] },
      options: {
        ...CHART_DEFAULTS,
        cutout: "65%",
        plugins: { legend: { position: "bottom", labels: { color: "#94a3b8", padding: 14 } } },
      },
    });

    // Gráfico de barras de actividad (simulado con historial reciente)
    cargarGraficoActividad();
  } catch (e) { console.error("Error estadísticas:", e); }
}

async function cargarGraficoActividad() {
  try {
    const res  = await fetch(`${API}/api/historial?limite=20`);
    const data = await res.json();
    if (!data.ok) return;
    // Agrupar por día
    const porDia = {};
    data.datos.forEach(item => {
      const d = (item.created_at || "").substring(0, 10);
      if (d) porDia[d] = (porDia[d] || 0) + 1;
    });
    const keys = Object.keys(porDia).sort().slice(-7);
    const vals = keys.map(k => porDia[k]);

    if (chartAct) chartAct.destroy();
    chartAct = new Chart(document.getElementById("chartActividad"), {
      type: "bar",
      data: {
        labels: keys.map(k => { const d = new Date(k + "T00:00:00"); return d.toLocaleDateString("es-PE", { day: "2-digit", month: "short" }); }),
        datasets: [{ label: "Imágenes analizadas", data: vals, backgroundColor: "rgba(99,102,241,.7)", borderRadius: 6, borderSkipped: false }],
      },
      options: {
        ...CHART_DEFAULTS,
        scales: {
          x: { ticks: { color: "#94a3b8" }, grid: { color: "#1e3a5f33" } },
          y: { ticks: { color: "#94a3b8", stepSize: 1 }, grid: { color: "#1e3a5f33" } },
        },
        plugins: { legend: { display: false } },
      },
    });
  } catch (e) { console.error("Error actividad:", e); }
}

// ─────────────────────────────────────────────────────────
// DASHBOARD — Galería de imágenes recientes
// ─────────────────────────────────────────────────────────
async function cargarHistorial() {
  try {
    const res  = await fetch(`${API}/api/historial?limite=12`);
    const data = await res.json();
    const gallery = document.getElementById("imgGallery");
    if (!data.ok || !data.datos.length) {
      gallery.innerHTML = `<div class="empty-state"><i class="bi bi-images"></i><p>Sin imágenes procesadas aún</p></div>`;
      return;
    }
    gallery.innerHTML = data.datos.map(img => {
      const cls = img.categoria_nombre || "sano";
      const conf = img.confianza ? `${(img.confianza * 100).toFixed(1)}%` : "—";
      return `
        <div class="gallery-item" onclick="abrirModal('/static/${img.ruta_archivo}')">
          <img src="/static/${img.ruta_archivo}" alt="${cls}" onerror="this.src='/static/img/placeholder.png'" />
          <div class="gallery-item-info">
            <span class="gallery-badge pill pill-${cls}">${cls}</span>
            <p class="gallery-conf"><i class="bi bi-patch-check"></i> ${conf}</p>
          </div>
        </div>`;
    }).join("");
  } catch (e) { console.error("Error galería:", e); }
}

// ─────────────────────────────────────────────────────────
// INFERENCIA — Análisis de imagen y documentos
// ─────────────────────────────────────────────────────────
let archivoSeleccionado = null;
let documentoSeleccionado = null;
let modoInferenciaActivo = "imagen";

// Alternar entre modo imagen y modo documento
window.switchModoInferencia = function(modo) {
  modoInferenciaActivo = modo;
  
  const tabImg = document.getElementById("subtab-imagen");
  const tabDoc = document.getElementById("subtab-documento");
  const panelImg = document.getElementById("panel-carga-imagen");
  const panelDoc = document.getElementById("panel-carga-documento");
  
  if (modo === "imagen") {
    tabImg.classList.add("active");
    tabDoc.classList.remove("active");
    panelImg.classList.remove("hidden");
    panelDoc.classList.add("hidden");
  } else {
    tabImg.classList.remove("active");
    tabDoc.classList.add("active");
    panelImg.classList.add("hidden");
    panelDoc.classList.remove("hidden");
  }
};

// ── MANEJO DE IMAGEN INDIVIDUAL ──
const dropZone    = document.getElementById("dropZone");
const inputImagen = document.getElementById("inputImagen");
const previewImg  = document.getElementById("previewImg");
const btnAnalizar = document.getElementById("btnAnalizar");

document.getElementById("btnSeleccionar").addEventListener("click", () => inputImagen.click());
inputImagen.addEventListener("change", () => { if (inputImagen.files[0]) seleccionarArchivo(inputImagen.files[0]); });

dropZone.addEventListener("click", e => {
  if (e.target !== inputImagen) inputImagen.click();
});
dropZone.addEventListener("dragover", e => { e.preventDefault(); dropZone.classList.add("drag-over"); });
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
dropZone.addEventListener("drop", e => {
  e.preventDefault(); dropZone.classList.remove("drag-over");
  const f = e.dataTransfer.files[0];
  if (f && f.type.startsWith("image/")) seleccionarArchivo(f);
  else toast("Solo se aceptan imágenes", "warn");
});

function seleccionarArchivo(file) {
  archivoSeleccionado = file;
  const url = URL.createObjectURL(file);
  previewImg.src = url;
  previewImg.classList.remove("hidden");
  document.getElementById("dropZoneContent").classList.add("hidden");
  btnAnalizar.disabled = false;
}

document.getElementById("btnAnalizar").addEventListener("click", async () => {
  if (!archivoSeleccionado) return;
  btnAnalizar.disabled = true;
  btnAnalizar.innerHTML = `<span class="spinner-border spinner-border-sm"></span> Analizando...`;
  const resultBody = document.getElementById("resultBody");
  resultBody.innerHTML = `<div class="empty-state"><i class="bi bi-cpu" style="animation:spin 1s linear infinite"></i><p>Procesando imagen...</p></div>`;

  try {
    const fd = new FormData();
    fd.append("imagen", archivoSeleccionado);
    const res  = await fetch(`${API}/api/inferencia`, { method: "POST", body: fd });
    const data = await res.json();

    if (!data.ok) throw new Error(data.error);

    const colores = { bache: "#ef4444", fisura: "#f59e0b", sano: "#10b981" };
    const color   = colores[data.clase] || "#6366f1";
    const probs   = data.probabilidades || {};

    resultBody.innerHTML = `
      <div style="text-align:center; margin-bottom:24px;">
        <div class="result-clase" style="color:${color}">${data.clase.toUpperCase()}</div>
        <div class="result-conf"><i class="bi bi-patch-check-fill" style="color:${color}"></i> Confianza: <strong style="color:${color}">${data.confianza_pct}%</strong></div>
        <div style="width:160px;height:8px;background:#1e293b;border-radius:99px;margin:0 auto 4px;">
          <div style="width:${data.confianza_pct}%;height:100%;background:${color};border-radius:99px;transition:width .8s;"></div>
        </div>
      </div>
      <h4 style="font-size:13px;color:var(--text-muted);margin-bottom:14px;">Probabilidades por clase</h4>
      ${Object.entries(probs).map(([cls, p]) => `
        <div class="prob-row">
          <span class="prob-label">${cls}</span>
          <div class="progress-bar-wrap" style="flex:1">
            <div class="progress-bar-fill" style="width:${(p*100).toFixed(1)}%;background:${colores[cls]||'#6366f1'}"></div>
          </div>
          <span class="prob-pct">${(p*100).toFixed(1)}%</span>
        </div>`).join("")}
        
      <div id="feedbackContainer" style="margin-top:20px;padding-top:16px;border-top:1px solid var(--border-solid);text-align:center">
        <p style="font-size:13px;margin-bottom:12px;color:var(--text-secondary)">¿Es correcta esta predicción?</p>
        <div style="display:flex;gap:10px;justify-content:center;flex-wrap:wrap">
            <button class="btn btn-sm btn-outline" style="color:#10b981;border-color:#10b981" onclick="enviarFeedback(${data.id}, '${data.clase}')">
                <i class="bi bi-hand-thumbs-up"></i> Sí, es correcto
            </button>
            <div style="position:relative;display:inline-block">
                <button class="btn btn-sm btn-outline" style="color:#ef4444;border-color:#ef4444" onclick="document.getElementById('feedbackDropdown').classList.toggle('hidden')">
                    <i class="bi bi-hand-thumbs-down"></i> No, corregir <i class="bi bi-chevron-down"></i>
                </button>
                <div id="feedbackDropdown" class="hidden" style="position:absolute;bottom:100%;left:50%;transform:translateX(-50%);margin-bottom:8px;background:var(--bg-card);border:1px solid var(--border-solid);border-radius:8px;padding:8px;z-index:10;display:flex;flex-direction:column;gap:4px;min-width:150px;box-shadow:0 4px 12px rgba(0,0,0,0.3)">
                    <div style="font-size:11px;color:var(--text-muted);padding:4px;text-align:left">Elegir correcta:</div>
                    ${['bache', 'fisura', 'sano'].filter(c => c !== data.clase).map(c => `
                        <button class="btn btn-sm btn-ghost" style="text-align:left;width:100%;text-transform:capitalize" onclick="enviarFeedback(${data.id}, '${c}')">${c}</button>
                    `).join('')}
                </div>
            </div>
        </div>
      </div>
    `;
    toast(`Clasificado como <strong>${data.clase}</strong> con ${data.confianza_pct}% de confianza`, "success");
    cargarEstadisticas();
  } catch (e) {
    resultBody.innerHTML = `<div class="empty-state" style="color:var(--danger)"><i class="bi bi-x-octagon"></i><p>${e.message}</p></div>`;
    toast(e.message, "error");
  } finally {
    btnAnalizar.disabled = false;
    btnAnalizar.innerHTML = `<i class="bi bi-cpu"></i> Analizar con IA`;
  }
});

// ─────────────────────────────────────────────────────────
// FEEDBACK DE INFERENCIA
// ─────────────────────────────────────────────────────────
async function enviarFeedback(id_imagen, clase_correcta) {
  try {
    const res = await fetch(`${API}/api/inferencia/feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id_imagen, clase_correcta })
    });
    const data = await res.json();
    if (!data.ok) throw new Error(data.error);
    
    toast(data.mensaje, "success");
    
    // Ocultar botones de feedback
    const container = document.getElementById("feedbackContainer");
    if (container) {
      container.innerHTML = `<div style="color:#10b981;font-size:13px"><i class="bi bi-check-circle-fill"></i> ¡Gracias por ayudar a mejorar el modelo!</div>`;
    }
  } catch (e) {
    toast(`Error al enviar feedback: ${e.message}`, "error");
  }
}

// ── MANEJO DE DOCUMENTOS PDF / DOCX ──
const dropZoneDoc     = document.getElementById("dropZoneDoc");
const inputDocumento  = document.getElementById("inputDocumento");
const previewDoc      = document.getElementById("previewDoc");
const btnSeleccionarDoc = document.getElementById("btnSeleccionarDoc");
const btnAnalizarDoc  = document.getElementById("btnAnalizarDoc");

if (btnSeleccionarDoc) btnSeleccionarDoc.addEventListener("click", () => inputDocumento.click());
if (inputDocumento) inputDocumento.addEventListener("change", () => { if (inputDocumento.files[0]) seleccionarDocumento(inputDocumento.files[0]); });

if (dropZoneDoc) {
  dropZoneDoc.addEventListener("click", e => {
    if (e.target !== inputDocumento) inputDocumento.click();
  });
  dropZoneDoc.addEventListener("dragover", e => { e.preventDefault(); dropZoneDoc.classList.add("drag-over"); });
  dropZoneDoc.addEventListener("dragleave", () => dropZoneDoc.classList.remove("drag-over"));
  dropZoneDoc.addEventListener("drop", e => {
    e.preventDefault(); dropZoneDoc.classList.remove("drag-over");
    const f = e.dataTransfer.files[0];
    if (f && (f.name.endsWith(".pdf") || f.name.endsWith(".docx"))) seleccionarDocumento(f);
    else toast("Solo se aceptan archivos PDF o DOCX", "warn");
  });
}

function seleccionarDocumento(file) {
  documentoSeleccionado = file;
  
  const docNameEl = document.getElementById("docName");
  if (docNameEl) docNameEl.textContent = file.name;
  
  let sizeKB = (file.size / 1024).toFixed(1);
  let sizeText = sizeKB > 1024 ? `${(sizeKB / 1024).toFixed(2)} MB` : `${sizeKB} KB`;
  
  const docSizeEl = document.getElementById("docSize");
  if (docSizeEl) docSizeEl.textContent = sizeText;
  
  if (previewDoc) previewDoc.classList.remove("hidden");
  
  const dropZoneDocContent = document.getElementById("dropZoneDocContent");
  if (dropZoneDocContent) dropZoneDocContent.classList.add("hidden");
  
  if (btnAnalizarDoc) btnAnalizarDoc.disabled = false;
}

if (btnAnalizarDoc) {
  btnAnalizarDoc.addEventListener("click", async () => {
    if (!documentoSeleccionado) return;
    btnAnalizarDoc.disabled = true;
    btnAnalizarDoc.innerHTML = `<span class="spinner-border spinner-border-sm"></span> Analizando documento...`;
    
    const resultBody = document.getElementById("resultBody");
    resultBody.innerHTML = `<div class="empty-state"><i class="bi bi-file-earmark-bar-graph" style="animation:spin 1s linear infinite; display: inline-block;"></i><p>Procesando documento... Extrayendo y clasificando imágenes.</p></div>`;

    try {
      const fd = new FormData();
      fd.append("documento", documentoSeleccionado);
      
      const res = await fetch(`${API}/api/inferencia-documento`, { method: "POST", body: fd });
      const data = await res.json();

      if (!data.ok) throw new Error(data.error);

      renderizarReporteDocumento(data);
      toast(`Análisis de documento completado. ${data.total_imagenes} imágenes analizadas.`, "success");
      cargarEstadisticas();
    } catch (e) {
      resultBody.innerHTML = `<div class="empty-state" style="color:var(--danger)"><i class="bi bi-x-octagon"></i><p>${e.message}</p></div>`;
      toast(e.message, "error");
    } finally {
      btnAnalizarDoc.disabled = false;
      btnAnalizarDoc.innerHTML = `<i class="bi bi-file-earmark-bar-graph"></i> Analizar Documento`;
    }
  });
}

function renderizarReporteDocumento(data) {
  const resultBody = document.getElementById("resultBody");
  const colores = { bache: "#ef4444", fisura: "#f59e0b", sano: "#10b981" };

  if (data.total_imagenes === 0) {
    resultBody.innerHTML = `
      <div class="empty-state" style="color: var(--text-secondary)">
        <i class="bi bi-file-earmark-x"></i>
        <h4 style="margin-top: 10px; font-weight: 600;">Sin imágenes válidas</h4>
        <p style="font-size: 13px; max-width: 280px; text-align: center;">${data.mensaje}</p>
      </div>
    `;
    return;
  }

  resultBody.innerHTML = `
    <div class="doc-report-container">
      <div class="doc-report-header">
        <div class="doc-title-row">
          <i class="bi bi-file-earmark-bar-graph-fill doc-icon"></i>
          <div>
            <h3 class="doc-report-name">${data.nombre_original}</h3>
            <span class="doc-report-meta">${data.total_imagenes} imágenes analizadas</span>
          </div>
        </div>
        <div class="doc-status-badge" style="background:${data.color_estado}18; color:${data.color_estado}; border:1px solid ${data.color_estado}35">
          <span class="status-badge-dot" style="background:${data.color_estado}"></span>
          ${data.estado_general}
        </div>
      </div>
      
      <p class="doc-diagnostico">${data.diagnostico}</p>
      
      <div class="doc-metrics-grid">
        <div class="doc-chart-box">
          <h4 class="metrics-subtitle"><i class="bi bi-pie-chart-fill"></i> Daño Dominante (% Imgs)</h4>
          <div class="doc-chart-canvas-wrapper">
            <canvas id="chartDocConteo" height="150"></canvas>
          </div>
        </div>
        <div class="doc-chart-box">
          <h4 class="metrics-subtitle"><i class="bi bi-shield-shaded"></i> Severidad Promedio</h4>
          <div style="display:flex; flex-direction:column; gap:10px; justify-content: center; height: calc(100% - 28px);">
            ${Object.entries(data.distribucion_severidad).map(([cls, pct]) => `
              <div class="prob-row" style="margin-bottom:0">
                <span class="prob-label" style="text-transform:capitalize; width:60px;">${cls}</span>
                <div class="progress-bar-wrap" style="flex:1">
                  <div class="progress-bar-fill" style="width:${pct}%; background:${colores[cls]}"></div>
                </div>
                <span class="prob-pct" style="width:40px;">${pct}%</span>
              </div>
            `).join("")}
          </div>
        </div>
      </div>
      
      <h4 class="metrics-subtitle mt-4" style="margin-bottom:8px;"><i class="bi bi-images"></i> Imágenes Extraídas y Clasificadas</h4>
      <div class="doc-extracted-grid">
        ${data.imagenes_analizadas.map(img => `
          <div class="doc-extracted-card" onclick="abrirModal('/static/${img.ruta_imagen.replace('static/', '')}')">
            <div class="doc-extracted-img-wrap">
              <img src="/static/${img.ruta_imagen.replace('static/', '')}" alt="${img.clase}" onerror="this.src='/static/img/placeholder.png'" />
              <span class="doc-page-badge">Pág. ${img.pagina}</span>
            </div>
            <div class="doc-extracted-body">
              <div class="doc-extracted-class">
                <span class="pill pill-${img.clase}">${img.clase}</span>
              </div>
              <div class="doc-extracted-conf">
                <i class="bi bi-patch-check-fill" style="color:${colores[img.clase]}"></i> ${img.confianza_pct}%
              </div>
            </div>
          </div>
        `).join("")}
      </div>
    </div>
  `;

  // Renderizar gráfico de torta dinámico para el documento
  setTimeout(() => {
    const ctx = document.getElementById("chartDocConteo");
    if (!ctx) return;
    
    const labels = Object.keys(data.distribucion_conteo);
    const vals   = Object.values(data.distribucion_conteo);
    const bgColors = labels.map(c => colores[c] || "#6366f1");

    if (chartDoc) chartDoc.destroy();
    
    chartDoc = new Chart(ctx, {
      type: "doughnut",
      data: {
        labels: labels.map(l => l.toUpperCase()),
        datasets: [{
          data: vals,
          backgroundColor: bgColors,
          borderColor: "#111827",
          borderWidth: 2,
          hoverOffset: 4
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: "right",
            labels: {
              color: "#94a3b8",
              font: { family: "Inter", size: 10 },
              boxWidth: 10,
              padding: 6
            }
          }
        },
        cutout: "60%"
      }
    });
  }, 100);
}


// ─────────────────────────────────────────────────────────
// ENTRENAMIENTO CNN
// ─────────────────────────────────────────────────────────
let trainEventSource = null;
const trainEpochsData = { accuracy: [], val_accuracy: [], loss: [], val_loss: [] };

document.getElementById("btnEntrenar").addEventListener("click", async () => {
  const epochs     = parseInt(document.getElementById("trainEpochs").value) || 15;
  const batch_size = parseInt(document.getElementById("trainBatch").value) || 32;
  const usar_demo  = document.querySelector('input[name="fuenteDatos"]:checked').value === "demo";

  const btn = document.getElementById("btnEntrenar");
  btn.disabled = true;
  btn.innerHTML = `<i class="bi bi-hourglass-split"></i> Iniciando...`;

  // Reiniciar datos de gráficos
  Object.keys(trainEpochsData).forEach(k => trainEpochsData[k] = []);

  const container = document.getElementById("trainProgress");
  container.innerHTML = `
    <div class="epoch-counter" id="epochCounter">Preparando entrenamiento...</div>
    <div class="progress-main"><div class="progress-main-fill" id="progressFill" style="width:0%"></div></div>
    <div class="train-metrics-row">
      <div class="train-metric"><div class="train-metric-val" id="tmAcc">—</div><div class="train-metric-lbl">Val Accuracy</div></div>
      <div class="train-metric"><div class="train-metric-val" id="tmLoss">—</div><div class="train-metric-lbl">Val Loss</div></div>
    </div>
  `;
  document.getElementById("trainMetricsChart").classList.remove("hidden");

  try {
    const res = await fetch(`${API}/api/entrenar`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ epochs, batch_size, usar_demo }),
    });
    const data = await res.json();
    if (!data.ok) throw new Error(data.error);
    toast("Entrenamiento iniciado en segundo plano", "info");

    // Conectar SSE para progreso
    if (trainEventSource) trainEventSource.close();
    trainEventSource = new EventSource(`${API}/api/entrenar/progreso`);
    trainEventSource.onmessage = e => {
      const ev = JSON.parse(e.data);
      actualizarProgressTrain(ev, epochs);
      if (ev.completado || ev.error) {
        trainEventSource.close();
        btn.disabled = false;
        btn.innerHTML = `<i class="bi bi-play-fill"></i> Iniciar Entrenamiento`;
        if (ev.completado) {
          toast(`✅ Entrenamiento completado — Accuracy: ${ev.accuracy}%`, "success");
          // Mostrar imagen de gráfico guardado
          const imgEl = document.getElementById("imgGraficoEntrenamiento");
          imgEl.src = `/static/graficos/entrenamiento_cnn_pavimento.png?t=${Date.now()}`;
          document.getElementById("trainGrafico").classList.remove("hidden");
        }
        if (ev.error) toast(`❌ Error: ${ev.error}`, "error");
      }
    };
    trainEventSource.onerror = () => trainEventSource.close();
  } catch (e) {
    toast(e.message, "error");
    btn.disabled = false;
    btn.innerHTML = `<i class="bi bi-play-fill"></i> Iniciar Entrenamiento`;
  }
});

function actualizarProgressTrain(ev, totalEpochs) {
  if (ev.en_curso === false && !ev.completado) return;
  const epoch    = ev.epoch || 0;
  const total    = ev.total_epochs || totalEpochs;
  const progreso = ev.progreso || Math.round((epoch / total) * 100);

  document.getElementById("progressFill").style.width = `${progreso}%`;
  document.getElementById("epochCounter").textContent = `Época ${epoch} / ${total} — ${progreso}%`;

  if (ev.accuracy !== undefined) document.getElementById("tmAcc").textContent  = `${ev.accuracy}%`;
  if (ev.loss !== undefined)     document.getElementById("tmLoss").textContent = ev.loss;

  // Actualizar Chart.js de entrenamiento en tiempo real
  if (ev.epoch && ev.accuracy !== undefined) {
    trainEpochsData.accuracy.push(ev.accuracy);
    trainEpochsData.loss.push(ev.loss || 0);

    if (!chartTrain) {
      chartTrain = new Chart(document.getElementById("chartEntrenamiento"), {
        type: "line",
        data: {
          labels: Array.from({ length: trainEpochsData.accuracy.length }, (_, i) => i + 1),
          datasets: [
            { label: "Accuracy (%)", data: trainEpochsData.accuracy, borderColor: "#6366f1", tension: .4, pointRadius: 3, fill: false },
            { label: "Loss", data: trainEpochsData.loss, borderColor: "#f59e0b", tension: .4, pointRadius: 3, fill: false, yAxisID: "yLoss" },
          ],
        },
        options: {
          animation: false,
          scales: {
            x: { ticks: { color: "#94a3b8" }, grid: { color: "#1e3a5f33" } },
            y:     { ticks: { color: "#6366f1" }, position: "left", grid: { color: "#1e3a5f33" } },
            yLoss: { ticks: { color: "#f59e0b" }, position: "right", grid: { display: false } },
          },
          plugins: { legend: { labels: { color: "#94a3b8" } } },
        },
      });
    } else {
      chartTrain.data.labels = Array.from({ length: trainEpochsData.accuracy.length }, (_, i) => i + 1);
      chartTrain.data.datasets[0].data = trainEpochsData.accuracy;
      chartTrain.data.datasets[1].data = trainEpochsData.loss;
      chartTrain.update();
    }
  }
}

// ─────────────────────────────────────────────────────────
// REGRESIÓN LINEAL
// ─────────────────────────────────────────────────────────
document.getElementById("btnEntrenarRegresion").addEventListener("click", async () => {
  const btn = document.getElementById("btnEntrenarRegresion");
  btn.disabled = true;
  btn.innerHTML = `<i class="bi bi-hourglass"></i> Entrenando...`;
  try {
    const res  = await fetch(`${API}/api/regresion/entrenar`, { method: "POST" });
    const data = await res.json();
    if (!data.ok) throw new Error(data.error);

    document.getElementById("regEcuacionTxt").textContent = data.ecuacion;
    document.getElementById("regMetrics").innerHTML = `
      <span class="reg-metric-chip">R² = ${data.r2}</span>
      <span class="reg-metric-chip">RMSE = ${data.rmse}</span>
    `;
    document.getElementById("regEcuacion").classList.remove("hidden");

    // Mostrar gráfico Matplotlib en base64
    if (data.img_base64) {
      const container = document.getElementById("regGraficoContainer");
      container.innerHTML = `<img src="data:image/png;base64,${data.img_base64}" style="width:100%;border-radius:10px;" alt="Gráfico de regresión" />`;
    }
    toast(`Regresión entrenada — R² = ${data.r2}`, "success");
  } catch (e) { toast(e.message, "error"); }
  finally {
    btn.disabled = false;
    btn.innerHTML = `<i class="bi bi-graph-up-arrow"></i> Entrenar Regresión`;
  }
});

document.getElementById("btnPredecir").addEventListener("click", async () => {
  const anios = parseFloat(document.getElementById("regAnios").value);
  if (isNaN(anios) || anios < 0) { toast("Ingresa un valor válido de años", "warn"); return; }
  try {
    const res  = await fetch(`${API}/api/regresion/predecir`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ anios }) });
    const data = await res.json();
    if (!data.ok) throw new Error(data.error);
    const el = document.getElementById("regResultado");
    el.style.borderColor = data.color;
    el.style.background  = `${data.color}15`;
    el.innerHTML = `
      <div class="pred-val" style="color:${data.color}">${data.indice_predicho} / 10</div>
      <div class="pred-lbl">Índice de deterioro predicho para <strong>${anios} años</strong></div>
      <div class="pred-estado" style="color:${data.color}">Estado: ${data.estado}</div>
    `;
    el.classList.remove("hidden");
  } catch (e) { toast(e.message, "error"); }
});

document.getElementById("btnAgregarDato").addEventListener("click", async () => {
  const anio   = parseInt(document.getElementById("nuevoDatoAnio").value);
  const indice = parseFloat(document.getElementById("nuevoDatoIndice").value);
  const zona   = document.getElementById("nuevoDatoZona").value || "Sin zona";
  if (isNaN(anio) || isNaN(indice)) { toast("Completa los campos requeridos", "warn"); return; }
  try {
    const res = await fetch(`${API}/api/regresion/datos`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ anio_pavimento: anio, indice_deterioro: indice, zona }) });
    const data = await res.json();
    if (!data.ok) throw new Error(data.error);
    toast("Dato agregado correctamente", "success");
    cargarDatosRegresion();
    document.getElementById("nuevoDatoAnio").value = "";
    document.getElementById("nuevoDatoIndice").value = "";
  } catch (e) { toast(e.message, "error"); }
});

async function cargarDatosRegresion() {
  try {
    const res  = await fetch(`${API}/api/regresion/datos`);
    const data = await res.json();
    const tbody = document.getElementById("tablaRegresionBody");
    if (!data.ok || !data.datos.length) {
      tbody.innerHTML = `<tr><td colspan="6" class="table-empty">Sin datos</td></tr>`;
      return;
    }
    tbody.innerHTML = data.datos.map(d => `
      <tr>
        <td><span style="color:var(--text-muted)">${d.id}</span></td>
        <td><strong>${d.anio_pavimento}</strong> años</td>
        <td><span style="color:var(--accent3)">${d.indice_deterioro?.toFixed(2) || "—"}</span></td>
        <td>${d.area_dano_m2 ? d.area_dano_m2.toFixed(1) + " m²" : "—"}</td>
        <td>${d.zona || "—"}</td>
        <td>${fmtFecha(d.created_at)}</td>
      </tr>`).join("");
  } catch (e) { console.error(e); }
}

// ─────────────────────────────────────────────────────────
// HISTORIAL (Tabla completa)
// ─────────────────────────────────────────────────────────
async function cargarTablaHistorial() {
  try {
    const res  = await fetch(`${API}/api/historial?limite=100`);
    const data = await res.json();
    const tbody = document.getElementById("tablaHistorialBody");
    if (!data.ok || !data.datos.length) {
      tbody.innerHTML = `<tr><td colspan="7" class="table-empty">Sin registros</td></tr>`;
      return;
    }
    tbody.innerHTML = data.datos.map(img => {
      const cls  = img.categoria_nombre || "—";
      const conf = img.confianza ? `${(img.confianza * 100).toFixed(1)}%` : "—";
      const dim  = img.ancho_px ? `${img.ancho_px}×${img.alto_px}` : "—";
      return `
        <tr>
          <td>${img.id}</td>
          <td><img class="thumb-img" src="/static/${img.ruta_archivo}" alt="${cls}" onerror="this.style.display='none'" onclick="abrirModal('/static/${img.ruta_archivo}')" /></td>
          <td>${img.nombre_original || img.nombre_archivo}</td>
          <td><span class="pill pill-${cls}">${cls}</span></td>
          <td><strong style="color:var(--accent2)">${conf}</strong></td>
          <td>${dim}</td>
          <td>${fmtFecha(img.created_at)}</td>
        </tr>`;
    }).join("");
  } catch (e) { console.error(e); }
}

// ─────────────────────────────────────────────────────────
// SESIONES DE ENTRENAMIENTO
// ─────────────────────────────────────────────────────────
async function cargarSesiones() {
  try {
    const res  = await fetch(`${API}/api/sesiones`);
    const data = await res.json();
    const tbody = document.getElementById("tablaSesionesBody");
    if (!data.ok || !data.datos.length) {
      tbody.innerHTML = `<tr><td colspan="8" class="table-empty">Sin sesiones registradas</td></tr>`;
      return;
    }
    const estadoPill = { completado: "pill-success", en_proceso: "pill-warning", error: "pill-danger" };
    tbody.innerHTML = data.datos.map(s => `
      <tr>
        <td>${s.id}</td>
        <td>${s.nombre_sesion || "—"}</td>
        <td>${s.epochs}</td>
        <td>${s.batch_size}</td>
        <td>${s.accuracy_final ? (s.accuracy_final * 100).toFixed(2) + "%" : "—"}</td>
        <td>${s.loss_final ? s.loss_final.toFixed(4) : "—"}</td>
        <td><span class="pill ${estadoPill[s.estado] || 'pill-info'}">${s.estado}</span></td>
        <td>${fmtFecha(s.created_at)}</td>
      </tr>`).join("");
  } catch (e) { console.error(e); }
}

// ─────────────────────────────────────────────────────────
// MODAL imagen ampliada
// ─────────────────────────────────────────────────────────
function abrirModal(src) {
  document.getElementById("modalImg").src = src;
  document.getElementById("modalOverlay").classList.remove("hidden");
}
document.getElementById("modalClose").addEventListener("click", () => document.getElementById("modalOverlay").classList.add("hidden"));
document.getElementById("modalOverlay").addEventListener("click", e => {
  if (e.target === document.getElementById("modalOverlay")) document.getElementById("modalOverlay").classList.add("hidden");
});

// ─────────────────────────────────────────────────────────
// Inicialización al cargar la página
// ─────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  cargarEstadisticas();
  cargarHistorial();
  // Actualizar KPIs cada 30 segundos
  setInterval(cargarEstadisticas, 30000);
  // Configurar drag-and-drop de dataset
  configurarDropZonesDataset();
  // Cargar info del dataset si se va a esa pestaña
  document.querySelectorAll(".nav-item").forEach(item => {
    if (item.dataset.tab === "entrenamiento") {
      item.addEventListener("click", () => { setTimeout(cargarInfoDataset, 100); });
    }
  });
});

// ─────────────────────────────────────────────────────────
// DATASET — Carga de imágenes por clase
// ─────────────────────────────────────────────────────────
const CLASES_DATASET = ["bache", "sano", "fisura"];

function seleccionarDataset(clase) {
  document.getElementById(`input${clase.charAt(0).toUpperCase() + clase.slice(1)}`).click();
}

function configurarDropZonesDataset() {
  CLASES_DATASET.forEach(clase => {
    const dz    = document.getElementById(`dz${clase.charAt(0).toUpperCase() + clase.slice(1)}`);
    const input = document.getElementById(`input${clase.charAt(0).toUpperCase() + clase.slice(1)}`);
    if (!dz || !input) return;

    // Click en zona de arrastre
    dz.addEventListener("click", () => input.click());

    // Drag & drop
    dz.addEventListener("dragover",  e => { e.preventDefault(); dz.classList.add("drag-over"); });
    dz.addEventListener("dragleave", () => dz.classList.remove("drag-over"));
    dz.addEventListener("drop", e => {
      e.preventDefault(); dz.classList.remove("drag-over");
      const files = Array.from(e.dataTransfer.files).filter(f => f.type.startsWith("image/"));
      if (files.length) subirImagenesDataset(clase, files);
      else toast("Solo se aceptan archivos de imagen", "warn");
    });

    // Selección via input
    input.addEventListener("change", () => {
      if (input.files.length) subirImagenesDataset(clase, Array.from(input.files));
    });
  });
}

async function subirImagenesDataset(clase, archivos) {
  if (!archivos.length) return;

  const pb    = document.getElementById("uploadProgressBar");
  const fill  = document.getElementById("uploadFill");
  const txt   = document.getElementById("uploadProgressTxt");
  const pct   = document.getElementById("uploadPct");

  pb.classList.remove("hidden");
  txt.textContent = `Subiendo ${archivos.length} imagen(es) de '${clase}'...`;
  fill.style.width = "10%";
  pct.textContent  = "10%";

  try {
    const fd = new FormData();
    fd.append("clase", clase);
    archivos.forEach(f => fd.append("imagenes", f));

    // Simular progreso visual durante la subida
    let prog = 10;
    const intervalo = setInterval(() => {
      prog = Math.min(prog + 8, 85);
      fill.style.width = `${prog}%`;
      pct.textContent  = `${prog}%`;
    }, 300);

    const res  = await fetch(`${API}/api/dataset/subir`, { method: "POST", body: fd });
    const data = await res.json();
    clearInterval(intervalo);

    fill.style.width = "100%";
    pct.textContent  = "100%";

    if (!data.ok) throw new Error(data.error);

    toast(`${data.mensaje}`, "success");
    await cargarInfoDataset();
  } catch (e) {
    toast(`Error subiendo imágenes: ${e.message}`, "error");
  } finally {
    setTimeout(() => pb.classList.add("hidden"), 1500);
  }
}

async function cargarInfoDataset() {
  try {
    const res  = await fetch(`${API}/api/dataset/info`);
    const data = await res.json();
    if (!data.ok) return;

    const clases = data.clases || {};
    const body   = document.getElementById("datasetSummaryBody");

    // Actualizar contadores en las cards
    CLASES_DATASET.forEach(clase => {
      const info     = clases[clase] || { train: 0, val: 0 };
      const total    = info.train + info.val;
      const countEl  = document.getElementById(`count${clase.charAt(0).toUpperCase() + clase.slice(1)}`);
      const statsEl  = document.getElementById(`stats${clase.charAt(0).toUpperCase() + clase.slice(1)}`);
      if (countEl) countEl.textContent = `${total} imgs`;
      if (statsEl) statsEl.innerHTML = `<span class="stat-chip train">Train: ${info.train}</span><span class="stat-chip val">Val: ${info.val}</span>`;
    });

    // Actualizar label de arquitectura
    const clasesActivas = Object.keys(clases).filter(c => (clases[c].train + clases[c].val) >= 5);
    const archOutput = document.getElementById("archOutput");
    if (archOutput) {
      if (clasesActivas.length > 0) {
        archOutput.textContent = `Softmax(${clasesActivas.length}): ${clasesActivas.join(" | ")}`;
      } else {
        archOutput.textContent = "Softmax(N clases)";
      }
    }

    if (data.total === 0) {
      body.innerHTML = `<div class="empty-state"><i class="bi bi-folder"></i><p>Aún no hay imágenes cargadas. Sube tus imágenes arriba.</p></div>`;
      return;
    }

    // Tabla resumen
    body.innerHTML = `
      <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:16px;">
        ${Object.entries(clases).map(([clase, info]) => {
          const total = info.train + info.val;
          const pct   = total >= 10 ? Math.round(info.train / total * 100) : 0;
          const colores = { bache:"#ef4444", sano:"#10b981", fisura:"#f59e0b" };
          const color  = colores[clase] || "#6366f1";
          return `
            <div style="background:var(--bg-surface);border-radius:10px;padding:16px;border:1px solid var(--border-solid)">
              <div style="font-size:13px;font-weight:600;color:${color};margin-bottom:10px;text-transform:capitalize">${clase}</div>
              <div style="font-size:28px;font-weight:800;color:var(--text-primary)">${total}</div>
              <div style="font-size:11px;color:var(--text-muted);margin-bottom:10px">imágenes totales</div>
              <div style="background:var(--bg-card);border-radius:99px;height:6px;overflow:hidden">
                <div style="width:${pct}%;height:100%;background:${color};border-radius:99px;transition:width .5s"></div>
              </div>
              <div style="display:flex;justify-content:space-between;margin-top:6px;font-size:11px;color:var(--text-muted)">
                <span>Train: <strong style="color:${color}">${info.train}</strong></span>
                <span>Val: <strong style="color:#f59e0b">${info.val}</strong></span>
              </div>
            </div>`;
        }).join("")}
      </div>
      <div style="margin-top:12px;font-size:12px;color:var(--text-muted);text-align:center">
        <i class="bi bi-info-circle"></i> Total: <strong style="color:var(--text-primary)">${data.total}</strong> imágenes listas para entrenar
      </div>
    `;
  } catch (e) { console.error("Error dataset info:", e); }
}

