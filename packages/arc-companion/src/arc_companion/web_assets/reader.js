(function () {
  "use strict";

  const snapshot = window.__ARC_COMPANION_SNAPSHOT__;
  const app = document.getElementById("reader-app");
  const main = document.getElementById("reader-main");
  const sidebar = document.getElementById("chapter-sidebar");
  const toggle = document.getElementById("sidebar-toggle");
  if (!snapshot || !app || !main || !sidebar || !toggle) return;

  const isChinese = String(snapshot.language || "").toLowerCase().startsWith("zh");
  const labels = isChinese ? {
    contents: "目录", guide: "章节导读", motivation: "动机", main_content: "主要内容",
    section_logic: "章节逻辑", book_position: "全文位置", prerequisites: "预备知识",
    explanation: "解释", prior_work: "此前工作", later_work: "后续工作",
    waiting: "尚在生成，刷新页面后将显示已通过校验的内容。", translationPending: "译文生成中",
    companionPending: "伴读生成中", glossary: "全文词汇表", status: "生成状态"
  } : {
    contents: "Contents", guide: "Chapter guide", motivation: "Motivation", main_content: "Main content",
    section_logic: "Section logic", book_position: "Position", prerequisites: "Prerequisites",
    explanation: "Explanation", prior_work: "Prior work", later_work: "Later work",
    waiting: "Generation is in progress. Refresh to show newly accepted material.", translationPending: "Translation pending",
    companionPending: "Companion pending", glossary: "Glossary", status: "Build status"
  };
  const chapterNodes = new Map();
  const chapterData = new Map((snapshot.chapters || []).map(chapter => [chapter.chapter_id, chapter]));
  const readingPositionKey = `arc-reader-anchor:${safeId(snapshot.paper_id || snapshot.title || "reader")}`;
  let readingObserver = null;

  restoreSidebar();
  toggle.addEventListener("click", () => setSidebar(!app.classList.contains("sidebar-collapsed")));
  renderNavigation();
  renderMain();
  installScrollSpy();
  restoreReadingPosition();

  function restoreSidebar() {
    let open = true;
    try { open = localStorage.getItem("arc-reader-sidebar") !== "closed"; } catch (_) { /* file privacy mode */ }
    setSidebar(open, false);
  }

  function setSidebar(open, remember = true) {
    app.classList.toggle("sidebar-collapsed", !open);
    toggle.setAttribute("aria-expanded", String(open));
    if (remember) {
      try { localStorage.setItem("arc-reader-sidebar", open ? "open" : "closed"); } catch (_) { /* ignore */ }
    }
  }

  function renderNavigation() {
    const heading = element("h2", labels.contents);
    const list = document.createElement("ol");
    sidebar.replaceChildren(heading, list);
    (snapshot.chapters || []).forEach((chapter, index) => {
      const item = document.createElement("li");
      const link = element("a", chapter.title || chapterLabel(index));
      link.href = `#${chapterAnchor(chapter.chapter_id)}`;
      link.dataset.chapter = chapter.chapter_id;
      link.addEventListener("click", event => {
        event.preventDefault();
        mountChapter(chapter.chapter_id);
        rememberAnchor(chapterAnchor(chapter.chapter_id));
        document.getElementById(chapterAnchor(chapter.chapter_id))?.scrollIntoView({behavior: "smooth", block: "start"});
        if (window.matchMedia("(max-width: 760px)").matches) setSidebar(false);
      });
      item.append(link);
      if (!(chapter.segments || []).length) item.append(element("div", labels.waiting, "pending"));
      list.append(item);
    });
  }

  function renderMain() {
    const header = document.createElement("header");
    header.className = "paper-header";
    header.append(element("h1", snapshot.title || "Companion Reader"));
    if ((snapshot.authors || []).length) header.append(element("p", snapshot.authors.join(", "), "authors"));
    const status = document.createElement("div");
    status.className = "build-status";
    status.append(element("span", `${labels.status}: ${snapshot.status || "preparing"}`, "status-pill"));
    const coverage = snapshot.coverage || {};
    status.append(element("span", `${(coverage.annotation_segment_ids || []).length}/${(coverage.segment_ids || []).length} companion`, "status-pill"));
    header.append(status);
    main.append(header);
    renderGlossary();
    const chapters = snapshot.chapters || [];
    if (!chapters.length) {
      main.append(element("div", labels.waiting, "empty-reader"));
      return;
    }
    chapters.forEach((chapter, index) => {
      const holder = document.createElement("section");
      holder.className = "chapter chapter-placeholder";
      holder.id = chapterAnchor(chapter.chapter_id);
      holder.dataset.chapter = chapter.chapter_id;
      holder.append(element("h2", chapter.title || chapterLabel(index)));
      holder.append(element("p", labels.waiting));
      chapterNodes.set(chapter.chapter_id, holder);
      main.append(holder);
      if (index < 2) mountChapter(chapter.chapter_id);
    });
    lazyMountChapters();
  }

  function renderGlossary() {
    if (!(snapshot.glossary || []).length) return;
    const details = document.createElement("details");
    details.className = "glossary";
    details.append(element("summary", labels.glossary));
    const list = document.createElement("dl");
    list.className = "glossary-list";
    snapshot.glossary.forEach(entry => {
      list.append(element("dt", entry.source || ""));
      list.append(element("dd", entry.target || "", "glossary-target"));
      list.append(element("dd", entry.explanation || "", "glossary-explanation"));
    });
    details.append(list);
    main.append(details);
  }

  function lazyMountChapters() {
    if (!("IntersectionObserver" in window)) return;
    const observer = new IntersectionObserver(entries => {
      entries.forEach(entry => {
        if (!entry.isIntersecting) return;
        mountChapter(entry.target.dataset.chapter);
        observer.unobserve(entry.target);
      });
    }, {rootMargin: "800px 0px"});
    chapterNodes.forEach(node => { if (!node.dataset.mounted) observer.observe(node); });
  }

  function mountChapter(chapterId) {
    const node = chapterNodes.get(chapterId);
    const chapter = chapterData.get(chapterId);
    if (!node || !chapter || node.dataset.mounted) return;
    node.dataset.mounted = "true";
    node.classList.remove("chapter-placeholder");
    const index = (snapshot.chapters || []).findIndex(item => item.chapter_id === chapterId);
    const title = element("h2", chapter.title || chapterLabel(index));
    node.replaceChildren(title);
    renderGuide(node, chapter.guide || []);
    if (!(chapter.segments || []).length) node.append(element("p", labels.waiting, "pending-layer"));
    (chapter.segments || []).forEach((segment, segmentIndex) => node.append(renderSegment(segment, segmentIndex)));
    typeset(node);
    observeReadingTargets(node);
  }

  function renderGuide(parent, guide) {
    if (!guide.length) return;
    const box = document.createElement("div");
    box.className = "chapter-guide";
    box.setAttribute("aria-label", labels.guide);
    guide.forEach(item => {
      const row = document.createElement("div");
      row.className = "guide-row";
      row.append(element("span", labels[item.kind] || item.kind, "guide-label"));
      appendRuns(row, item.runs || []);
      box.append(row);
    });
    parent.append(box);
  }

  function renderSegment(segment, index) {
    const unit = document.createElement("article");
    unit.className = "reading-unit";
    unit.id = segmentAnchor(segment.segment_id);
    const grid = document.createElement("div");
    grid.className = "unit-grid";
    const source = layer("source-layer");
    if (segment.title) source.append(element("h3", segment.title, "segment-title"));
    (segment.source || []).forEach(block => source.append(renderSourceBlock(block)));
    const side = document.createElement("div");
    side.className = "side-layers";
    if (snapshot.translation_mode !== "skipped") side.append(renderTranslation(segment.translation));
    side.append(renderCompanion(segment.companion));
    grid.append(source, side);
    unit.append(grid);
    return unit;
  }

  function renderSourceBlock(block) {
    const wrapper = document.createElement("div");
    wrapper.className = `source-block ${safeClass(block.kind)}`;
    if (block.kind === "equation" || block.kind === "math" || block.kind === "display_math") {
      if (block.number) wrapper.append(element("span", block.number, "equation-number"));
      appendRuns(wrapper, block.math || block.runs || []);
    } else if (block.kind === "figure" || block.kind === "image") {
      const figure = document.createElement("figure");
      figure.className = "source-figure";
      (block.assets || []).forEach(asset => {
        const image = document.createElement("img");
        image.src = safeAssetUrl(asset.url);
        image.alt = block.caption || "";
        image.loading = "lazy";
        figure.append(image);
      });
      if (block.caption) figure.append(element("figcaption", block.caption));
      wrapper.append(figure);
    } else if (block.kind === "table") {
      if (block.caption) wrapper.append(element("div", block.caption, "table-caption"));
      const table = document.createElement("table");
      table.className = "source-table";
      (block.rows || []).forEach(row => {
        const tr = document.createElement("tr");
        row.forEach(cell => tr.append(element("td", cell)));
        table.append(tr);
      });
      wrapper.append(table);
    } else if ((block.items || []).length) {
      const list = document.createElement(block.ordered ? "ol" : "ul");
      block.items.forEach(item => list.append(element("li", item)));
      wrapper.append(list);
    } else {
      appendRuns(wrapper, block.runs || []);
    }
    return wrapper;
  }

  function renderTranslation(translation) {
    const box = layer("translation-layer");
    if (!translation) {
      box.append(element("div", labels.translationPending, "pending-layer"));
      return box;
    }
    (translation.blocks || []).forEach(block => {
      const row = document.createElement("div");
      row.className = "translated-block";
      appendRuns(row, block.runs || []);
      box.append(row);
    });
    return box;
  }

  function renderCompanion(companion) {
    const box = layer("companion-layer");
    if (!companion) {
      box.append(element("div", labels.companionPending, "pending-layer"));
      return box;
    }
    (companion.sections || []).forEach(section => {
      box.append(element("span", labels[section.kind] || section.kind, "annotation-label"));
      if (section.claims) {
        const list = document.createElement("ul");
        list.className = "claim-list";
        section.claims.forEach(claim => {
          const item = document.createElement("li");
          appendRuns(item, claim.runs || []);
          appendSources(item, claim.sources || []);
          list.append(item);
        });
        box.append(list);
      } else {
        const body = document.createElement("div");
        appendRuns(body, section.runs || []);
        appendSources(body, section.sources || []);
        box.append(body);
      }
    });
    return box;
  }

  function appendRuns(parent, runs) {
    runs.forEach(run => {
      if (run.type === "math") {
        const math = document.createElement(run.display ? "div" : "span");
        math.className = run.display ? "math-display" : "math-inline";
        math.dataset.tex = String(run.tex || "");
        math.dataset.display = run.display ? "true" : "false";
        parent.append(math);
      } else if (run.type === "link") {
        const href = safeHref(run.href);
        if (!href) parent.append(document.createTextNode(String(run.text || "")));
        else {
          const link = element("a", run.text || href);
          link.href = href;
          if (!href.startsWith("#")) { link.target = "_blank"; link.rel = "noopener noreferrer"; }
          parent.append(link);
        }
      } else {
        parent.append(document.createTextNode(String(run.text || "")));
      }
    });
  }

  function appendSources(parent, sources) {
    if (!sources.length) return;
    const line = document.createElement("div");
    line.className = "sources";
    sources.forEach((source, index) => {
      if (index) line.append(document.createTextNode("; "));
      const href = safeHref(source.url, true);
      if (!href) return;
      const link = element("a", source.title || href);
      link.href = href;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      line.append(link, document.createTextNode(` — ${source.locator || ""}`));
    });
    parent.append(line);
  }

  function typeset(root) {
    root.querySelectorAll("[data-tex]").forEach(node => {
      if (!window.katex) {
        node.textContent = node.dataset.tex || "";
        node.classList.add("math-error");
        return;
      }
      try {
        window.katex.render(node.dataset.tex || "", node, {
          displayMode: node.dataset.display === "true", throwOnError: false,
          strict: "warn", trust: false, output: "htmlAndMathml"
        });
      } catch (_) {
        node.textContent = node.dataset.tex || "";
        node.classList.add("math-error");
      }
    });
  }

  function installScrollSpy() {
    if (!("IntersectionObserver" in window)) return;
    const visible = new Map();
    readingObserver = new IntersectionObserver(entries => {
      entries.forEach(entry => {
        if (entry.isIntersecting) visible.set(entry.target.id, entry);
        else visible.delete(entry.target.id);
      });
      const candidates = Array.from(visible.values());
      const segmentCandidates = candidates.filter(entry => entry.target.classList.contains("reading-unit"));
      const current = (segmentCandidates.length ? segmentCandidates : candidates)
        .sort((a, b) => Math.abs(a.boundingClientRect.top) - Math.abs(b.boundingClientRect.top))[0];
      if (!current) return;
      const chapter = current.target.classList.contains("chapter") ? current.target : current.target.closest(".chapter");
      const chapterId = chapter?.dataset.chapter || "";
      sidebar.querySelectorAll("a[data-chapter]").forEach(link => link.classList.toggle("is-current", link.dataset.chapter === chapterId));
      rememberAnchor(current.target.id);
    }, {rootMargin: "-15% 0px -70%"});
    chapterNodes.forEach(node => observeReadingTargets(node));
  }

  function observeReadingTargets(chapter) {
    if (!readingObserver || !chapter) return;
    readingObserver.observe(chapter);
    chapter.querySelectorAll(".reading-unit").forEach(node => readingObserver.observe(node));
  }

  function rememberAnchor(anchor) {
    if (!anchor || !document.getElementById(anchor)) return;
    try { history.replaceState(history.state, "", `#${anchor}`); } catch (_) { /* file privacy mode */ }
    try { localStorage.setItem(readingPositionKey, anchor); } catch (_) { /* file privacy mode */ }
  }

  function restoreReadingPosition() {
    let anchor = String(location.hash || "").replace(/^#/, "");
    if (!anchor) {
      try { anchor = localStorage.getItem(readingPositionKey) || ""; } catch (_) { /* file privacy mode */ }
    }
    if (!anchor) return;
    const chapter = (snapshot.chapters || []).find(item => {
      if (chapterAnchor(item.chapter_id) === anchor) return true;
      return (item.segments || []).some(segment => segmentAnchor(segment.segment_id) === anchor);
    });
    if (!chapter) return;
    mountChapter(chapter.chapter_id);
    requestAnimationFrame(() => requestAnimationFrame(() => {
      document.getElementById(anchor)?.scrollIntoView({behavior: "auto", block: "start"});
    }));
  }

  function layer(name) {
    const node = document.createElement("section");
    node.className = `layer ${name}`;
    return node;
  }
  function element(tag, text, className) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    node.textContent = String(text || "");
    return node;
  }
  function chapterLabel(index) { return isChinese ? `第 ${index + 1} 章` : `Chapter ${index + 1}`; }
  function chapterAnchor(id) { return `chapter-${safeId(id)}`; }
  function segmentAnchor(id) { return `segment-${safeId(id)}`; }
  function safeId(value) { return String(value || "").replace(/[^A-Za-z0-9_.:-]/g, "-"); }
  function safeClass(value) { return String(value || "text").replace(/[^A-Za-z0-9_-]/g, "-"); }
  function safeAssetUrl(value) {
    const text = String(value || "");
    return /^assets\/[A-Za-z0-9_./-]+$/.test(text) && !text.includes("..") ? text : "";
  }
  function safeHref(value, externalOnly = false) {
    const text = String(value || "").trim();
    if (!externalOnly && /^#[A-Za-z0-9_.:-]+$/.test(text)) return text;
    try {
      const parsed = new URL(text);
      return parsed.protocol === "http:" || parsed.protocol === "https:" ? parsed.href : "";
    } catch (_) { return ""; }
  }
})();
