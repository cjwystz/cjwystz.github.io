(() => {
  "use strict";
  const visits = document.querySelector("[data-site-visits]");
  const callback = "cjwVisits" + Math.random().toString(36).slice(2);
  const script = document.createElement("script");
  let settled = false;
  const finish = (value) => {
    if (settled) return;
    settled = true;
    clearTimeout(timer);
    visits.textContent = value;
    script.remove();
    // Keep a harmless callback for a response that arrives after the timeout.
    window[callback] = () => {};
  };
  window[callback] = (data) => {
    const count = data && data.site_pv;
    finish(Number.isSafeInteger(count) && count >= 0 ? count.toLocaleString("zh-CN") : "暂不可用");
  };
  script.src = "https://busuanzi.ibruce.info/busuanzi?jsonpCallback=" + callback;
  script.referrerPolicy = "origin";
  script.onerror = () => finish("暂不可用");
  const timer = setTimeout(() => finish("暂不可用"), 12000);
  document.head.appendChild(script);

  const button = document.querySelector("[data-like-url]");
  const countLabel = button.querySelector("[data-like-count]");
  const status = document.querySelector("[data-like-status]");
  const query = "?url=" + encodeURIComponent(button.dataset.likeUrl);
  const request = async (update = false) => {
    const response = await fetch("https://applause.chabouis.fr/" + (update ? "update-claps" : "get-claps") + query, {
      method: update ? "POST" : "GET",
      ...(update ? { headers: { "Content-Type": "text/plain" }, body: JSON.stringify("1,4.0.0") } : {}),
      referrerPolicy: "origin",
      signal: AbortSignal.timeout(12000),
    });
    if (!response.ok) throw new Error("Counter request failed");
    const text = (await response.text()).trim();
    if (!/^\d+$/.test(text)) throw new Error("Invalid count");
    const count = Number(text);
    if (!Number.isSafeInteger(count)) throw new Error("Invalid count");
    return count;
  };
  let currentCount;
  const show = (count) => {
    currentCount = count;
    countLabel.textContent = count.toLocaleString("zh-CN");
  };
  request().then(show).catch(() => { countLabel.textContent = "暂不可用"; }).finally(() => { button.disabled = false; });
  button.addEventListener("click", async () => {
    button.disabled = true;
    status.textContent = "保存中…";
    try {
      const previous = currentCount;
      const count = await request(true);
      show(count);
      status.textContent = previous !== undefined && count <= previous ? "服务暂未增加计数，请稍后再试" : "谢谢支持！";
    } catch {
      status.textContent = "未确认保存成功，请稍后刷新查看";
    } finally {
      button.disabled = false;
    }
  });
})();
