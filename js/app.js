(() => {
    'use strict';

    // ── Tab Switching ──
    const tabs = document.querySelectorAll('.nav-tab');
    const tabContents = document.querySelectorAll('.tab-content');

    function switchTab(tabName) {
        tabs.forEach(t => t.classList.toggle('active', t.dataset.tab === tabName));
        tabContents.forEach(tc => tc.classList.toggle('active', tc.id === `tab-${tabName}`));
        window.location.hash = tabName;
    }

    tabs.forEach(tab => {
        tab.addEventListener('click', () => switchTab(tab.dataset.tab));
    });

    // ── Math + Markdown Rendering ──
    // Protect LaTeX from marked by replacing with placeholders, then render with KaTeX after
    function renderMathInMarkdown(src) {
        const mathBlocks = [];
        let idx = 0;

        // Protect display math ($$...$$) first — must come before inline
        src = src.replace(/\$\$([\s\S]+?)\$\$/g, (_, tex) => {
            const placeholder = `%%MATH_BLOCK_${idx}%%`;
            mathBlocks.push({ placeholder, tex: tex.trim(), display: true });
            idx++;
            return placeholder;
        });

        // Protect inline math ($...$) — avoid matching escaped \$ or empty $$
        src = src.replace(/(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)/g, (_, tex) => {
            const placeholder = `%%MATH_BLOCK_${idx}%%`;
            mathBlocks.push({ placeholder, tex: tex.trim(), display: false });
            idx++;
            return placeholder;
        });

        // Run marked on the math-free markdown
        let html = marked.parse(src);

        // Replace placeholders with KaTeX-rendered HTML
        for (const { placeholder, tex, display } of mathBlocks) {
            try {
                const rendered = katex.renderToString(tex, {
                    displayMode: display,
                    throwOnError: false,
                    trust: true,
                    strict: false
                });
                html = html.replace(placeholder, rendered);
            } catch {
                // Fallback: show raw LaTeX in a code block
                const escaped = tex.replace(/</g, '&lt;').replace(/>/g, '&gt;');
                html = html.replace(placeholder, `<code>${escaped}</code>`);
            }
        }

        return html;
    }

    function stripFrontmatter(text) {
        const match = text.match(/^---\n([\s\S]*?)\n---\n/);
        if (!match) return { meta: null, body: text };
        const metaBlock = match[1];
        const body = text.slice(match[0].length);
        // Simple YAML-like parser for our frontmatter
        const meta = {};
        let currentKey = null;
        let currentList = null;
        let currentObj = null;
        for (const line of metaBlock.split('\n')) {
            const topLevel = line.match(/^(\w+):\s*(.*)/);
            if (topLevel) {
                if (currentKey && currentList) meta[currentKey] = currentList;
                currentList = null;
                currentObj = null;
                const [, key, val] = topLevel;
                currentKey = key;
                if (val === '' || val === '[]') {
                    currentList = [];
                } else {
                    meta[key] = val.replace(/^["']|["']$/g, '');
                    currentKey = null;
                }
            } else if (line.match(/^\s+-\s+date:/)) {
                currentObj = { date: line.match(/date:\s*(.*)/)[1].trim() };
                currentList.push(currentObj);
            } else if (line.match(/^\s+note:/) && currentObj) {
                currentObj.note = line.match(/note:\s*(.*)/)[1].trim();
            } else if (line.match(/^\s+-\s/)) {
                const val = line.match(/^\s+-\s+(.*)/)[1].replace(/^["']|["']$/g, '');
                if (currentList) currentList.push(val);
            }
        }
        if (currentKey && currentList) meta[currentKey] = currentList;
        return { meta, body };
    }

    async function loadMarkdown(path) {
        const res = await fetch(`${path}?t=${Date.now()}`, { cache: 'no-store' });
        if (!res.ok) throw new Error(`Failed to load ${path}`);
        const text = await res.text();
        const { meta, body } = stripFrontmatter(text);
        const html = renderMathInMarkdown(body);
        // Prepend metadata banner if we have frontmatter
        if (meta && meta.date) {
            let banner = `<div class="note-meta-banner"><span class="note-published">Published: ${meta.date}</span>`;
            if (meta.updates && meta.updates.length > 0) {
                banner += '<div class="note-updates"><strong>Updates:</strong><ul>';
                for (const u of meta.updates) {
                    banner += `<li><span class="note-update-date">${u.date}</span> — ${u.note}</li>`;
                }
                banner += '</ul></div>';
            }
            banner += '</div>';
            return banner + html;
        }
        return html;
    }

    // ── About Page ──
    async function initAbout() {
        const container = document.getElementById('about-content');
        try {
            container.innerHTML = await loadMarkdown('content/about.md');
        } catch {
            container.innerHTML = '<p>Failed to load profile content.</p>';
        }
    }

    // ── Reading Notes ──
    const notesLayout = document.querySelector('.notes-layout');
    const readerPane = document.getElementById('reader-pane');
    const readerContent = document.getElementById('reader-content');
    const readerClose = document.getElementById('reader-close');
    const notesList = document.getElementById('notes-list');
    let activeCard = null;

    function openReader(html) {
        readerContent.innerHTML = html;
        readerPane.classList.add('open');
        notesLayout.classList.add('reader-open');
    }

    function closeReader() {
        readerPane.classList.remove('open');
        notesLayout.classList.remove('reader-open');
        if (activeCard) {
            activeCard.classList.remove('active');
            activeCard = null;
        }
    }

    readerClose.addEventListener('click', closeReader);

    async function loadNote(file, card) {
        if (activeCard) activeCard.classList.remove('active');
        activeCard = card;
        card.classList.add('active');

        readerContent.innerHTML = '<p class="loading">Loading...</p>';
        readerPane.classList.add('open');
        notesLayout.classList.add('reader-open');

        try {
            const html = await loadMarkdown(`notes/${file}`);
            openReader(html);
        } catch {
            openReader('<p>Failed to load this note.</p>');
        }
    }

    async function initNotes() {
        try {
            const res = await fetch(`notes/index.json?t=${Date.now()}`, { cache: 'no-store' });
            const notes = await res.json();

            notesList.innerHTML = '';
            // Sort notes by date descending
            notes.sort((a, b) => b.date.localeCompare(a.date));
            // Group by year
            const byYear = {};
            notes.forEach(note => {
                const year = note.date.slice(0, 4);
                if (!byYear[year]) byYear[year] = [];
                byYear[year].push(note);
            });
            // Render year sections (most recent first)
            Object.keys(byYear).sort((a, b) => b - a).forEach(year => {
                const section = document.createElement('div');
                section.className = 'notes-year-section';
                const heading = document.createElement('h3');
                heading.className = 'notes-year-heading';
                heading.textContent = year;
                section.appendChild(heading);

                byYear[year].forEach(note => {
                    const card = document.createElement('div');
                    card.className = 'note-card';
                    card.innerHTML = `
                        <div class="note-card-title">${note.title}</div>
                        <div class="note-card-meta">
                            <span>${note.date}</span>
                            ${note.tags.map(t => `<span class="note-card-tag">${t}</span>`).join('')}
                        </div>
                    `;
                    card.addEventListener('click', () => loadNote(note.file, card));
                    section.appendChild(card);
                });
                notesList.appendChild(section);
            });
        } catch {
            notesList.innerHTML = '<p>Failed to load notes list.</p>';
        }
    }

    // ── Init ──
    function init() {
        initAbout();
        initNotes();

        // Handle initial hash
        const hash = window.location.hash.replace('#', '');
        if (hash === 'notes' || hash === 'about') {
            switchTab(hash);
        }
    }

    // Handle hash changes (back/forward navigation)
    window.addEventListener('hashchange', () => {
        const hash = window.location.hash.replace('#', '');
        if (hash === 'notes' || hash === 'about') {
            switchTab(hash);
        }
    });

    init();
})();
