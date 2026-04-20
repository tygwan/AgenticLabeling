// Review Workspace — 3 pane: source list (left), viewer (center), inspector (right)

const ReviewWorkspace = ({ sources, settings, onUpdateObject, panelOrder }) => {
  const [activeId, setActiveId] = React.useState(sources[0]?.id);
  const [selectedObj, setSelectedObj] = React.useState(null);
  const [filter, setFilter] = React.useState('all'); // all | pending | validated | failed
  const [show, setShow] = React.useState({ box: true, mask: true, label: true });
  const [localObjects, setLocalObjects] = React.useState(() => {
    const m = {};
    sources.forEach(s => { m[s.id] = s.objects.map(o => ({ ...o })); });
    return m;
  });

  const source = sources.find(s => s.id === activeId) || sources[0];
  const objects = localObjects[source.id] || [];
  const visibleObjs = objects;
  const current = visibleObjs.find(o => o.object_id === selectedObj) || null;

  // Filter sources
  const filteredSources = sources.filter(s => {
    if (filter === 'all') return true;
    if (filter === 'pending') return s.status === 'pending' || s.status === 'in_review';
    if (filter === 'validated') return s.status === 'validated';
    if (filter === 'failed') return s.status === 'failed';
    return true;
  });

  const setValidated = (oid, v) => {
    setLocalObjects(prev => ({
      ...prev,
      [source.id]: prev[source.id].map(o => o.object_id === oid ? { ...o, validated: v } : o),
    }));
  };

  const nextObj = React.useCallback(() => {
    const pending = objects.filter(o => o.validated == null);
    if (pending.length === 0) return;
    const idx = pending.findIndex(o => o.object_id === selectedObj);
    const next = pending[(idx + 1 + pending.length) % pending.length] || pending[0];
    setSelectedObj(next.object_id);
  }, [objects, selectedObj]);

  const prevObj = React.useCallback(() => {
    const pending = objects.filter(o => o.validated == null);
    if (pending.length === 0) return;
    const idx = pending.findIndex(o => o.object_id === selectedObj);
    const next = pending[(idx - 1 + pending.length) % pending.length] || pending[pending.length - 1];
    setSelectedObj(next.object_id);
  }, [objects, selectedObj]);

  // Keyboard
  React.useEffect(() => {
    const onKey = (e) => {
      if (e.target.matches('input, textarea, select')) return;
      const k = e.key.toLowerCase();
      if (k === 'j') { e.preventDefault(); nextObj(); }
      else if (k === 'k') { e.preventDefault(); prevObj(); }
      else if (k === 'a' && selectedObj) { e.preventDefault(); setValidated(selectedObj, 'approved'); nextObj(); }
      else if (k === 'd' && selectedObj) { e.preventDefault(); setValidated(selectedObj, 'deleted'); nextObj(); }
      else if (k === 'u' && selectedObj) { e.preventDefault(); setValidated(selectedObj, null); }
      else if (k === 'b') { e.preventDefault(); setShow(s => ({ ...s, box: !s.box })); }
      else if (k === 'm') { e.preventDefault(); setShow(s => ({ ...s, mask: !s.mask })); }
      else if (k === 'l') { e.preventDefault(); setShow(s => ({ ...s, label: !s.label })); }
      else if (k === 'arrowright') {
        e.preventDefault();
        const idx = filteredSources.findIndex(s => s.id === source.id);
        const next = filteredSources[(idx + 1) % filteredSources.length];
        setActiveId(next.id); setSelectedObj(null);
      } else if (k === 'arrowleft') {
        e.preventDefault();
        const idx = filteredSources.findIndex(s => s.id === source.id);
        const next = filteredSources[(idx - 1 + filteredSources.length) % filteredSources.length];
        setActiveId(next.id); setSelectedObj(null);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [selectedObj, source?.id, filteredSources, nextObj, prevObj]);

  // Stats
  const stats = {
    total: objects.length,
    approved: objects.filter(o => o.validated === 'approved').length,
    deleted: objects.filter(o => o.validated === 'deleted').length,
    pending: objects.filter(o => o.validated == null).length,
  };

  const leftPane = (
    <SourceListPane
      sources={filteredSources}
      activeId={activeId}
      onPick={(id) => { setActiveId(id); setSelectedObj(null); }}
      filter={filter}
      setFilter={setFilter}
      allCounts={{
        all: sources.length,
        pending: sources.filter(s => s.status === 'pending' || s.status === 'in_review').length,
        validated: sources.filter(s => s.status === 'validated').length,
        failed: sources.filter(s => s.status === 'failed').length,
      }}
    />
  );
  const rightPane = (
    <InspectorPane
      source={source}
      objects={objects}
      stats={stats}
      selected={selectedObj}
      onSelect={setSelectedObj}
      current={current}
      onApprove={(oid) => { setValidated(oid, 'approved'); nextObj(); }}
      onDelete={(oid) => { setValidated(oid, 'deleted'); nextObj(); }}
      onReset={(oid) => setValidated(oid, null)}
      show={show}
      setShow={setShow}
      style={settings.objectListStyle}
    />
  );

  return (
    <div className={`review-grid order-${panelOrder}`}>
      {leftPane}
      <div className="review-center">
        <div className="viewer-toolbar">
          <div className="source-title">
            <span className="file-name">{source.file_name}</span>
            <span className="file-meta">{source.project}</span>
            <StatusPill status={source.status} />
          </div>
          <div className="viewer-toolbar-right">
            <OverlayToggle active={show.box}   onClick={() => setShow(s => ({ ...s, box: !s.box }))}   Icon={IconBox}   label="Boxes" kbd="B" />
            <OverlayToggle active={show.mask}  onClick={() => setShow(s => ({ ...s, mask: !s.mask }))} Icon={IconMask}  label="Masks" kbd="M" />
            <OverlayToggle active={show.label} onClick={() => setShow(s => ({ ...s, label: !s.label }))} Icon={IconLabel} label="Labels" kbd="L" />
          </div>
        </div>
        <ImageViewer
          source={source}
          objects={objects}
          selectedId={selectedObj}
          onSelect={setSelectedObj}
          show={show}
        />
        <div className="review-footbar">
          <div className="foot-left">
            <span className="foot-stat">
              <strong>{stats.approved}</strong> approved
            </span>
            <span className="foot-stat">
              <strong>{stats.deleted}</strong> deleted
            </span>
            <span className="foot-stat muted">
              <strong>{stats.pending}</strong> pending
            </span>
          </div>
          <div className="foot-right">
            <span className="foot-hint">
              <Kbd>J</Kbd><Kbd>K</Kbd> navigate · <Kbd>A</Kbd> approve · <Kbd>D</Kbd> delete · <Kbd>→</Kbd> next source
            </span>
          </div>
        </div>
      </div>
      {rightPane}
    </div>
  );
};

const OverlayToggle = ({ active, onClick, Icon, label, kbd }) => (
  <button className={`overlay-toggle ${active ? 'on' : ''}`} onClick={onClick} title={`${label} (${kbd})`}>
    <Icon size={14} />
    <span>{label}</span>
    <Kbd>{kbd}</Kbd>
  </button>
);

const SourceListPane = ({ sources, activeId, onPick, filter, setFilter, allCounts }) => {
  const FILTERS = [
    { id: 'all',       label: 'All' },
    { id: 'pending',   label: 'Pending' },
    { id: 'validated', label: 'Done' },
    { id: 'failed',    label: 'Failed' },
  ];
  return (
    <aside className="pane left-pane">
      <div className="pane-header">
        <div className="pane-title">Sources</div>
        <div className="pane-count">{sources.length}</div>
      </div>
      <div className="pane-filters">
        {FILTERS.map(f => (
          <button
            key={f.id}
            className={`filter-chip ${filter === f.id ? 'on' : ''}`}
            onClick={() => setFilter(f.id)}
          >
            {f.label}
            <span className="filter-count">{allCounts[f.id]}</span>
          </button>
        ))}
      </div>
      <div className="source-list">
        {sources.map(s => {
          const thumbImgs = s.objects.slice(0, 3).map(o => CATEGORY_COLORS[o.category]);
          return (
            <button
              key={s.id}
              className={`source-row ${activeId === s.id ? 'active' : ''}`}
              onClick={() => onPick(s.id)}
            >
              <div className="source-thumb">
                <img src={s.url} onError={(e) => { e.target.style.display='none'; }} alt="" />
                <div className="source-thumb-fallback">
                  <IconImage size={16} />
                </div>
              </div>
              <div className="source-meta">
                <div className="source-meta-name">{s.file_name}</div>
                <div className="source-meta-sub">
                  <StatusPill status={s.status} />
                  <span className="obj-count">{s.objects.length} obj</span>
                </div>
                <div className="source-classes">
                  {thumbImgs.slice(0, 3).map((c, i) => <span key={i} className="class-dot" style={{ background: c }} />)}
                  {s.classes.slice(0, 3).map(c => <span key={c} className="class-tag">{c}</span>)}
                  {s.classes.length > 3 && <span className="class-more">+{s.classes.length - 3}</span>}
                </div>
              </div>
            </button>
          );
        })}
      </div>
    </aside>
  );
};

const InspectorPane = ({ source, objects, stats, selected, onSelect, current, onApprove, onDelete, onReset, show, setShow, style }) => {
  const visible = objects.filter(o => o.validated !== 'deleted');
  return (
    <aside className="pane right-pane">
      <div className="pane-header">
        <div className="pane-title">Objects</div>
        <div className="pane-count">{visible.length}/{objects.length}</div>
      </div>
      <div className="stat-bar">
        <div className="stat-seg" style={{ flex: stats.approved || 0.001, background: 'var(--c-ok)' }} title={`${stats.approved} approved`} />
        <div className="stat-seg" style={{ flex: stats.pending || 0.001, background: 'var(--c-border-strong)' }} title={`${stats.pending} pending`} />
        <div className="stat-seg" style={{ flex: stats.deleted || 0.001, background: 'var(--c-err)', opacity: 0.5 }} title={`${stats.deleted} deleted`} />
      </div>
      <div className={`object-list style-${style}`}>
        {objects.map(o => (
          <ObjectRow
            key={o.object_id}
            o={o}
            active={o.object_id === selected}
            onClick={() => onSelect(o.object_id)}
            onApprove={() => onApprove(o.object_id)}
            onDelete={() => onDelete(o.object_id)}
            onReset={() => onReset(o.object_id)}
            style={style}
          />
        ))}
        {objects.length === 0 && (
          <div className="object-empty">
            <IconWarn size={20} />
            <div>No objects detected</div>
            {source.error && <div className="object-empty-msg">{source.error}</div>}
          </div>
        )}
      </div>
      {current && (
        <div className="inspector-detail">
          <div className="detail-title">Detail</div>
          <KV k="object_id"      v={current.object_id} mono />
          <KV k="category"       v={<><IconDot size={8} color={CATEGORY_COLORS[current.category]} />&nbsp;{current.category}</>} />
          <KV k="confidence"     v={`${(current.confidence * 100).toFixed(1)}%`} />
          <KV k="bbox (xywh)"    v={current.bbox.map(n => n.toFixed(3)).join(' · ')} mono />
          <KV k="mask_path"      v={`masks/${current.object_id}.png`} mono />
          <KV k="is_validated"   v={current.validated || '—'} />
        </div>
      )}
    </aside>
  );
};

const KV = ({ k, v, mono }) => (
  <div className="kv">
    <span className="kv-k">{k}</span>
    <span className={`kv-v ${mono ? 'mono' : ''}`}>{v}</span>
  </div>
);

const ObjectRow = ({ o, active, onClick, onApprove, onDelete, onReset, style }) => {
  const color = CATEGORY_COLORS[o.category] || '#64748b';
  const deleted = o.validated === 'deleted';
  const approved = o.validated === 'approved';
  return (
    <div
      className={`obj-row ${active ? 'active' : ''} ${deleted ? 'deleted' : ''} ${approved ? 'approved' : ''}`}
      onClick={onClick}
      style={{ '--obj-color': color }}
    >
      <div className="obj-swatch" />
      <div className="obj-body">
        <div className="obj-top">
          <span className="obj-cat">{o.category}</span>
          <span className="obj-conf">{Math.round(o.confidence * 100)}%</span>
        </div>
        <div className="obj-id">{o.object_id}</div>
      </div>
      <div className="obj-actions">
        {o.validated == null && (
          <>
            <button className="icon-btn ok" onClick={(e) => { e.stopPropagation(); onApprove(); }} title="Approve (A)"><IconCheck size={14} stroke={2.5} /></button>
            <button className="icon-btn err" onClick={(e) => { e.stopPropagation(); onDelete(); }} title="Delete (D)"><IconX size={14} stroke={2.5} /></button>
          </>
        )}
        {o.validated != null && (
          <button className="icon-btn" onClick={(e) => { e.stopPropagation(); onReset(); }} title="Reset (U)">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M1 4v6h6"/><path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10"/></svg>
          </button>
        )}
      </div>
    </div>
  );
};

Object.assign(window, { ReviewWorkspace });
