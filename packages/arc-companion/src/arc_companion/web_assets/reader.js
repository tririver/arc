(function () {
  "use strict";

  const snapshot = window.__ARC_COMPANION_SNAPSHOT__;
  const app = document.getElementById("reader-app");
  const main = document.getElementById("reader-main");
  const sidebar = document.getElementById("chapter-sidebar");
  const toggle = document.getElementById("sidebar-toggle");
  if (!snapshot || !app || !main || !sidebar || !toggle) return;

  const targetLanguage = String(snapshot.language || "und");
  const sourceLanguage = String(snapshot.source_language || "und");
  const targetDirection = String(snapshot.direction || "auto");
  const sourceDirection = String(snapshot.source_direction || "auto");
  if (document.documentElement) {
    document.documentElement.lang = targetLanguage;
    document.documentElement.dir = targetDirection;
  }

  const isChinese = String(snapshot.language || "").toLowerCase().startsWith("zh");
  const labels = isChinese ? {
    contents: "目录", guide: "章节导读", motivation: "动机", main_content: "主要内容",
    section_logic: "章节逻辑", prerequisites: "预备知识", pedagogical_comparison: "教材顺序比较",
    historical_context: "历史背景", supplementary_reading: "补充阅读",
    sidebarOpen: "收起侧栏", sidebarClosed: "展开侧栏",
    explanation: "解释", prior_work: "此前工作", later_work: "后续工作",
    waiting: "尚在生成，刷新页面后将显示已通过校验的内容。", translationPending: "译文生成中",
    companionPending: "伴读生成中", glossary: "全文术语表", status: "生成状态", sourceAppendix: "原文附录"
  } : {
    contents: "Contents", guide: "Chapter guide", motivation: "Motivation", main_content: "Main content",
    section_logic: "Section logic", prerequisites: "Prerequisites", pedagogical_comparison: "Pedagogical comparison",
    historical_context: "Historical context", supplementary_reading: "Further reading",
    sidebarOpen: "Collapse sidebar", sidebarClosed: "Expand sidebar",
    explanation: "Explanation", prior_work: "Prior work", later_work: "Later work",
    waiting: "Generation is in progress. Refresh to show newly accepted material.", translationPending: "Translation pending",
    companionPending: "Companion pending", glossary: "Glossary", status: "Build status", sourceAppendix: "Source appendix"
  };
  const chapterNodes = new Map();
  const chapterData = new Map((snapshot.chapters || []).map(chapter => [chapter.chapter_id, chapter]));
  const readingPositionKey = `arc-reader-anchor:${safeId(snapshot.paper_id || snapshot.title || "reader")}`;
  let readingObserver = null;
  let glossaryNode = null;

  restoreSidebar();
  toggle.addEventListener("click", () => setSidebar(app.classList.contains("sidebar-collapsed")));
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
    const toggleLabel = open ? labels.sidebarOpen : labels.sidebarClosed;
    toggle.textContent = toggleLabel;
    toggle.setAttribute("aria-label", toggleLabel);
    toggle.setAttribute("title", toggleLabel);
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
      if (!(chapter.segments || []).length && !(chapter.guide || []).length && !chapter.structural_only) {
        item.append(element("div", labels.waiting, "pending"));
      }
      list.append(item);
    });
    if (glossaryEnabled()) {
      const item = document.createElement("li");
      const link = element("a", labels.glossary);
      link.href = "#glossary";
      link.dataset.glossary = "true";
      link.addEventListener("click", event => {
        event.preventDefault();
        mountGlossary();
        rememberAnchor("glossary");
        requestAnimationFrame(() => document.getElementById("glossary")?.scrollIntoView({behavior: "smooth", block: "start"}));
        if (window.matchMedia("(max-width: 760px)").matches) setSidebar(false);
      });
      item.append(link);
      list.append(item);
    }
  }

  function renderMain() {
    const header = document.createElement("header");
    header.className = "paper-header";
    header.append(bilingualHeading(
      "h1", snapshot.source_title, snapshot.translated_title,
      snapshot.title || "Companion Reader"
    ));
    if ((snapshot.authors || []).length) header.append(element("p", snapshot.authors.join(", "), "authors"));
    const status = document.createElement("div");
    status.className = "build-status";
    status.append(element("span", `${labels.status}: ${snapshot.status || "preparing"}`, "status-pill"));
    const coverage = snapshot.coverage || {};
    status.append(element("span", `${(coverage.annotation_segment_ids || []).length}/${(coverage.segment_ids || []).length} companion`, "status-pill"));
    header.append(status);
    main.append(header);
    const chapters = snapshot.chapters || [];
    if (!chapters.length) {
      main.append(element("div", labels.waiting, "empty-reader"));
      if (!(snapshot.appendices || []).length && !glossaryEnabled()) return;
    }
    chapters.forEach((chapter, index) => {
      const holder = document.createElement("section");
      holder.className = "chapter chapter-placeholder";
      holder.id = chapterAnchor(chapter.chapter_id);
      holder.dataset.chapter = chapter.chapter_id;
      holder.append(bilingualHeading(
        "h2", chapter.source_title, chapter.translated_title,
        chapter.title || chapterLabel(index)
      ));
      holder.append(element("p", labels.waiting));
      chapterNodes.set(chapter.chapter_id, holder);
      main.append(holder);
      if (index < 2) mountChapter(chapter.chapter_id);
    });
    (snapshot.appendices || []).forEach(appendix => main.append(renderAppendix(appendix)));
    prepareGlossary();
    lazyMountChapters();
  }

  function glossaryEnabled() {
    return snapshot.translation_mode === "enabled" && (snapshot.glossary || []).length > 0;
  }

  function prepareGlossary() {
    if (!glossaryEnabled()) return;
    glossaryNode = document.createElement("section");
    glossaryNode.id = "glossary";
    glossaryNode.className = "glossary glossary-placeholder";
    glossaryNode.append(element("h2", labels.glossary));
    main.append(glossaryNode);
    if (String(location.hash || "") === "#glossary") mountGlossary();
    if (!("IntersectionObserver" in window)) { mountGlossary(); return; }
    const observer = new IntersectionObserver(entries => {
      if (!entries.some(entry => entry.isIntersecting)) return;
      mountGlossary();
      observer.disconnect();
    }, {rootMargin: "1000px 0px"});
    observer.observe(glossaryNode);
  }

  function mountGlossary() {
    if (!glossaryNode || glossaryNode.dataset.mounted) return;
    glossaryNode.dataset.mounted = "true";
    glossaryNode.classList.remove("glossary-placeholder");
    const heading = element("h2", labels.glossary);
    const list = document.createElement("dl");
    list.className = "glossary-list";
    snapshot.glossary.forEach(entry => {
      const source = document.createElement("dt");
      setLanguage(source, sourceLanguage, sourceDirection);
      source.append(termElement(entry.source || "", entry));
      const target = document.createElement("dd");
      target.className = "glossary-target";
      setLanguage(target, targetLanguage, targetDirection);
      target.append(termElement(entry.target || "", entry));
      list.append(source, target);
      list.append(element("dd", entry.explanation || "", "glossary-explanation"));
    });
    glossaryNode.replaceChildren(heading, list);
    observeReadingTargets(glossaryNode);
  }

  function renderAppendix(appendix) {
    const section = document.createElement("section");
    section.className = "source-appendix";
    section.id = `appendix-${safeId(appendix.appendix_id || "source")}`;
    section.append(bilingualHeading(
      "h2", appendix.source_title, appendix.translated_title,
      appendix.title || labels.sourceAppendix
    ));
    const source = layer("source-layer", sourceLanguage, sourceDirection);
    (appendix.source || []).forEach(block => source.append(renderSourceBlock(block)));
    section.append(source);
    typeset(section);
    return section;
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
    const title = bilingualHeading(
      "h2", chapter.source_title, chapter.translated_title,
      chapter.title || chapterLabel(index)
    );
    node.replaceChildren(title);
    renderGuide(node, chapter.guide || []);
    if (!(chapter.segments || []).length && !(chapter.guide || []).length && !chapter.structural_only) {
      node.append(element("p", labels.waiting, "pending-layer"));
    }
    (chapter.segments || []).forEach((segment, segmentIndex) => {
      if (segment.structural_only && !(segment.source || []).length) return;
      node.append(renderSegment(segment, segmentIndex));
    });
    typeset(node);
    observeReadingTargets(node);
  }

  function renderGuide(parent, guide) {
    if (!guide.length) return;
    const box = document.createElement("div");
    box.className = "chapter-guide";
    setLanguage(box, targetLanguage, targetDirection);
    box.setAttribute("aria-label", labels.guide);
    guide.forEach(item => {
      const row = document.createElement("div");
      row.className = "guide-row";
      row.append(element("span", labels[item.kind] || item.kind, "guide-label"));
      appendRuns(row, item.runs || []);
      appendSources(row, item.sources || []);
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
    const source = layer("source-layer", sourceLanguage, sourceDirection);
    (segment.source || []).forEach(block => source.append(renderSourceBlock(block)));
    const side = document.createElement("div");
    side.className = "side-layers";
    if (!segment.structural_only) {
      if (snapshot.translation_mode !== "skipped") side.append(renderTranslation(segment.translation));
      side.append(renderCompanion(segment.companion));
    }
    grid.append(source);
    if (side.childNodes.length) grid.append(side);
    else grid.classList.add("source-only");
    unit.append(grid);
    return unit;
  }

  function renderSourceBlock(block) {
    const wrapper = document.createElement("div");
    wrapper.className = `source-block ${safeClass(block.kind)}`;
    setLanguage(
      wrapper,
      String(block.language || sourceLanguage),
      String(block.direction || sourceDirection)
    );
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
    } else if (isStructuralKind(block.kind) && block.translated_title) {
      const level = block.kind === "subsubsection" ? "h5" : block.kind === "subsection" ? "h4" : "h3";
      const heading = document.createElement(level);
      const sourceTitle = element("span", "", "title-source");
      setLanguage(sourceTitle, sourceLanguage, sourceDirection);
      appendRuns(sourceTitle, block.runs || []);
      const translatedTitle = element("span", "", "title-translation");
      setLanguage(translatedTitle, targetLanguage, targetDirection);
      if ((block.translated_title_runs || []).length) appendRuns(translatedTitle, block.translated_title_runs);
      else translatedTitle.textContent = String(block.translated_title || "");
      heading.append(sourceTitle, translatedTitle);
      wrapper.append(heading);
    } else {
      appendRuns(wrapper, block.runs || []);
    }
    return wrapper;
  }

  function renderTranslation(translation) {
    const box = layer("translation-layer", targetLanguage, targetDirection);
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
    const box = layer("companion-layer", targetLanguage, targetDirection);
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
      } else if (run.type === "term") {
        parent.append(termElement(run.text || "", run));
      } else {
        parent.append(document.createTextNode(String(run.text || "")));
      }
    });
  }

  function termElement(text, entry) {
    const source = String(entry.source || "");
    const target = String(entry.target || "");
    if (!source || !target || source.normalize("NFKC").toLocaleLowerCase() === target.normalize("NFKC").toLocaleLowerCase()) {
      return document.createTextNode(String(text || ""));
    }
    const node = element("span", text, "glossary-term");
    const description = `${source} ↔ ${target}`;
    node.tabIndex = 0;
    node.dataset.tooltip = description;
    node.setAttribute("aria-label", description);
    return node;
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
      sidebar.querySelectorAll("a[data-glossary]").forEach(link => link.classList.toggle("is-current", current.target.id === "glossary"));
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
    if (anchor === "glossary" && glossaryEnabled()) mountGlossary();
    else if (chapter) mountChapter(chapter.chapter_id);
    else if (!document.getElementById(anchor)) return;
    requestAnimationFrame(() => requestAnimationFrame(() => {
      document.getElementById(anchor)?.scrollIntoView({behavior: "auto", block: "start"});
    }));
  }

  function layer(name, language, direction) {
    const node = document.createElement("section");
    node.className = `layer ${name}`;
    if (language) setLanguage(node, language, direction || "auto");
    return node;
  }
  function bilingualHeading(tag, sourceTitle, translatedTitle, fallback) {
    const source = String(sourceTitle || "").trim();
    const translated = String(translatedTitle || "").trim();
    const heading = document.createElement(tag);
    if (source && translated && source !== translated) {
      const sourceNode = element("span", source, "title-source");
      setLanguage(sourceNode, sourceLanguage, sourceDirection);
      const targetNode = element("span", translated, "title-translation");
      setLanguage(targetNode, targetLanguage, targetDirection);
      heading.append(sourceNode, targetNode);
    } else {
      heading.textContent = translated || source || String(fallback || "");
      setLanguage(
        heading,
        translated ? targetLanguage : source ? sourceLanguage : targetLanguage,
        translated ? targetDirection : source ? sourceDirection : targetDirection
      );
    }
    return heading;
  }
  function setLanguage(node, language, direction) {
    node.setAttribute("lang", String(language || "und"));
    node.setAttribute("dir", String(direction || "auto"));
  }
  function isStructuralKind(kind) {
    return ["part", "chapter", "heading", "section", "subsection", "subsubsection"].includes(String(kind || "").toLowerCase());
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
