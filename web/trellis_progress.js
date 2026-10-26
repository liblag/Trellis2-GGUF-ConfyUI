import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// Text-only status display. No serialized widgets or changes to workflow inputs.
app.registerExtension({
    name: "liblag.TrellisStageProgress",
    setup() {
        const panel = document.createElement("div");
        panel.style.cssText = "position:fixed;bottom:18px;right:18px;max-width:min(520px,80vw);z-index:10000;background:#181c24ed;color:#e5edf8;border:1px solid #596b82;border-radius:8px;padding:10px 14px;font:12px/1.5 sans-serif;white-space:pre-wrap;pointer-events:none;display:none";
        document.body.appendChild(panel);
        let latest = null;
        let received = 0;
        let hideTimer;
        api.addEventListener("trellis2-stage", ({ detail: d }) => {
            latest = d;
            received = Date.now();
            panel.style.display = "block";
            panel.style.borderColor = d.status.includes("error") ? "#ef7777" : "#596b82";
            clearTimeout(hideTimer);
            if (["done", "error"].includes(d.status)) hideTimer = setTimeout(() => { panel.style.display = "none"; }, 30000);
            render();
        });
        function render() {
            if (!latest) return;
            const d = latest;
            const since = Math.floor((Date.now() - received) / 1000);
            const count = d.total ? ` — ${d.step}/${d.total}` : "";
            const memory = d.rss_gb !== undefined ? `\nRAM процесса: ${d.rss_gb} ГБ · свободно: ${d.ram_available_gb} ГБ` : "";
            const age = since >= 30 && !["done", "error"].includes(d.status) ? `\nНет новых сообщений ${since} с — проверьте лог Colab и соединение.` : "";
            panel.textContent = `Trellis · нода ${d.node ?? "?"} · ${d.status}\n${d.stage}${count}\nВсего: ${d.elapsed} с · этап: ${d.stage_elapsed} с${memory}${d.error ? "\n" + d.error : ""}${age}`;
        }
        setInterval(render, 1000);
    },
});
