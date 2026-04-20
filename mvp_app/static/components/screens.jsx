// Export screen + Settings + Tweaks panel + Shortcuts help.

const ExportScreen = ({ workspace }) => {
  const [format, setFormat] = React.useState('yolo');
  const [datasetName, setDatasetName] = React.useState('mvp-dataset');
  const [splits, setSplits] = React.useState({ train: 80, val: 15, test: 5 });
  const [filterValidated, setFilterValidated] = React.useState(true);
  const [includeEmpty, setIncludeEmpty] = React.useState(false);
  const [exporting, setExporting] = React.useState(false);
  const [lastExport, setLastExport] = React.useState(null);
  const [exportError, setExportError] = React.useState(null);

  const sources = workspace?.sources || [];
  const totalImages = sources.length;
  const totalObjects = sources.reduce((n, s) => n + (s.objects?.length || 0), 0);
  const totalValidated = sources.reduce(
    (n, s) => n + (s.objects?.filter(o => o.validated === 'approved').length || 0), 0);
  const classCounts = {};
  sources.forEach(s => (s.objects || []).forEach(o => {
    if (filterValidated && o.validated !== 'approved') return;
    classCounts[o.category] = (classCounts[o.category] || 0) + 1;
  }));
  const totalClasses = Object.keys(classCounts).length;
  const sortedClassCounts = Object.entries(classCounts).sort((a, b) => b[1] - a[1]).slice(0, 10);
  const maxCount = sortedClassCounts[0]?.[1] || 1;
  const approxSize = '—';

  const onGenerate = async () => {
    setExporting(true); setExportError(null);
    try {
      const result = await API.exportDataset({
        datasetName, exportFormat: format, onlyValidated: filterValidated,
        splits,
      });
      setLastExport(result);
    } catch (e) {
      setExportError(String(e));
    } finally {
      setExporting(false);
    }
  };

  return (
    <div className="export-grid">
      <section className="export-config">
        <div className="card-eyebrow">Configure</div>
        <h2 className="card-title">Export dataset</h2>

        <div className="form-field">
          <label>Dataset name</label>
          <input value={datasetName} onChange={e => setDatasetName(e.target.value)} className="mono" />
        </div>

        <div className="form-field">
          <label>Format</label>
          <div className="format-grid">
            {[
              { id: 'yolo', name: 'YOLO', sub: 'Darknet / Ultralytics txt' },
              { id: 'coco', name: 'COCO', sub: 'Single instances_*.json' },
              { id: 'voc',  name: 'Pascal VOC', sub: 'Per-image xml', disabled: true },
            ].map(f => (
              <button
                key={f.id}
                className={`format-card ${format === f.id ? 'on' : ''} ${f.disabled ? 'disabled' : ''}`}
                onClick={() => !f.disabled && setFormat(f.id)}
                disabled={f.disabled}
              >
                <div className="fmt-name">{f.name}</div>
                <div className="fmt-sub">{f.sub}</div>
                {f.disabled && <div className="fmt-tag">Soon</div>}
              </button>
            ))}
          </div>
        </div>

        <div className="form-field">
          <label>Splits <span className="hint">{splits.train + splits.val + splits.test}%</span></label>
          <div className="splits">
            <SplitSlider label="train" color="var(--c-ok)"    v={splits.train} onChange={v => setSplits({ ...splits, train: v })} />
            <SplitSlider label="val"   color="var(--c-accent)" v={splits.val}   onChange={v => setSplits({ ...splits, val: v })} />
            <SplitSlider label="test"  color="var(--c-warn)"   v={splits.test}  onChange={v => setSplits({ ...splits, test: v })} />
          </div>
          <div className="split-bar">
            <div style={{ flex: splits.train, background: 'var(--c-ok)' }} />
            <div style={{ flex: splits.val, background: 'var(--c-accent)' }} />
            <div style={{ flex: splits.test, background: 'var(--c-warn)' }} />
          </div>
        </div>

        <div className="form-field">
          <label>Filters</label>
          <label className="check-row">
            <input type="checkbox" checked={filterValidated} onChange={e => setFilterValidated(e.target.checked)} />
            <span>Only validated objects <span className="hint">recommended</span></span>
          </label>
          <label className="check-row">
            <input type="checkbox" checked={includeEmpty} onChange={e => setIncludeEmpty(e.target.checked)} />
            <span>Include images with no objects</span>
          </label>
        </div>
      </section>

      <section className="export-summary">
        <div className="card-eyebrow">Summary</div>
        <h3 className="card-title sm">{datasetName}.zip</h3>

        <div className="summary-stats">
          <div className="sum-stat"><div className="sum-v">{(filterValidated ? totalValidated : totalObjects).toLocaleString()}</div><div className="sum-k">Objects</div></div>
          <div className="sum-stat"><div className="sum-v">{totalImages}</div><div className="sum-k">Images</div></div>
          <div className="sum-stat"><div className="sum-v">{totalClasses}</div><div className="sum-k">Classes</div></div>
          <div className="sum-stat"><div className="sum-v">{approxSize}</div><div className="sum-k">~Size</div></div>
        </div>

        <div className="file-tree">
          <div className="tree-line"><span className="tree-k">📦</span> {datasetName}.zip</div>
          <div className="tree-line indent"><span className="tree-k">├─</span> {format === 'yolo' ? 'data.yaml' : 'instances.json'}</div>
          <div className="tree-line indent"><span className="tree-k">├─</span> images/</div>
          <div className="tree-line indent2"><span className="tree-k">│&nbsp;&nbsp;├─</span> train/ <span className="tree-n">({Math.round(totalImages*splits.train/100)})</span></div>
          <div className="tree-line indent2"><span className="tree-k">│&nbsp;&nbsp;├─</span> val/ <span className="tree-n">({Math.round(totalImages*splits.val/100)})</span></div>
          <div className="tree-line indent2"><span className="tree-k">│&nbsp;&nbsp;└─</span> test/ <span className="tree-n">({Math.round(totalImages*splits.test/100)})</span></div>
          <div className="tree-line indent"><span className="tree-k">└─</span> labels/</div>
        </div>

        <div className="class-dist">
          <div className="dist-title">Class distribution{filterValidated ? ' (validated only)' : ''}</div>
          {sortedClassCounts.length === 0 ? (
            <div className="dist-row" style={{ color: 'var(--c-muted)' }}>No {filterValidated ? 'validated ' : ''}objects yet.</div>
          ) : sortedClassCounts.map(([cat, n]) => (
            <div key={cat} className="dist-row">
              <span className="dist-cat"><IconDot size={8} color={CATEGORY_COLORS[cat] || '#64748b'} /> {cat}</span>
              <div className="dist-bar"><div className="dist-fill" style={{ width: `${(n/maxCount)*100}%`, background: CATEGORY_COLORS[cat] || '#64748b' }} /></div>
              <span className="dist-n">{n}</span>
            </div>
          ))}
        </div>

        <button className="btn primary large" onClick={onGenerate} disabled={exporting || totalImages === 0}>
          <IconExport size={16} /> {exporting ? 'Generating…' : `Generate ${format.toUpperCase()} export`}
        </button>
        {exportError && <div className="export-hint" style={{ color: 'var(--c-err)' }}>{exportError}</div>}
        {lastExport && (
          <div className="export-hint">
            <IconCheck size={12} stroke={3} /> Ready: <a href={lastExport.download_url}>{lastExport.dataset_name}.zip</a> · {lastExport.image_count} images · {lastExport.object_count} objects
          </div>
        )}
      </section>
    </div>
  );
};

const SplitSlider = ({ label, v, onChange, color }) => (
  <div className="split-slider">
    <span className="split-label" style={{ color }}>{label}</span>
    <input type="range" min="0" max="100" value={v} onChange={e => onChange(+e.target.value)} />
    <span className="split-v">{v}%</span>
  </div>
);

const SettingsScreen = () => {
  const [health, setHealth] = React.useState(null);
  const [activeNav, setActiveNav] = React.useState('models');

  React.useEffect(() => {
    let cancelled = false;
    API.fetchHealth().then(h => { if (!cancelled) setHealth(h); }).catch(() => {});
    return () => { cancelled = true; };
  }, []);

  const backend = health?.segmentation_backend || 'unknown';
  const stats = health?.stats || { sources: 0, objects: 0, categories: 0 };

  return (
    <div className="settings-grid">
      <div className="settings-nav">
        <div className="settings-nav-title">Settings</div>
        {[['models','Models & runtime'], ['paths','Paths & storage'], ['workspace','Workspace defaults']].map(([id, label]) => (
          <button key={id} className={`settings-nav-item ${activeNav === id ? 'active' : ''}`} onClick={() => setActiveNav(id)}>{label}</button>
        ))}
      </div>
      <div className="settings-body">
        {activeNav === 'models' && <>
          <div className="settings-section">
            <h3>Detection backend</h3>
            <div className="setting-card">
              <div className="sc-left">
                <div className="sc-title">Florence-2 <span className="sc-ver">{health ? 'loaded' : '—'}</span></div>
                <div className="sc-sub">trust_remote_code; dtype from FLORENCE_DTYPE env</div>
              </div>
              <div className="sc-right">
                <StatusPill status={health ? 'running' : 'failed'} />
              </div>
            </div>
          </div>

          <div className="settings-section">
            <h3>Segmentation backend</h3>
            <div className="setting-card">
              <div className="sc-left">
                <div className="sc-title">SAM3 <span className="sc-ver">{backend}</span></div>
                <div className="sc-sub">Prompted with detection bboxes (cxcywh normalized)</div>
              </div>
              <div className="sc-right">
                <StatusPill status={backend === 'sam3' ? 'running' : (backend === 'box-fallback' ? 'failed' : 'queued')} />
              </div>
            </div>
          </div>
        </>}

        {activeNav === 'paths' && (
          <div className="settings-section">
            <h3>Registry</h3>
            <div className="setting-card">
              <div className="sc-left">
                <div className="sc-title"><IconDatabase size={14} />&nbsp;SQLite registry</div>
                <div className="sc-sub">{stats.sources?.toLocaleString() ?? 0} sources · {stats.objects?.toLocaleString() ?? 0} objects · {stats.categories ?? 0} categories</div>
              </div>
            </div>
          </div>
        )}

        {activeNav === 'workspace' && (
          <div className="settings-section">
            <h3>Workspace defaults</h3>
            <div className="setting-card">
              <div className="sc-left">
                <div className="sc-title">Theme, density, accent color</div>
                <div className="sc-sub">Persisted to localStorage. Press ? for shortcuts, click the Tweak icon in the side nav to adjust.</div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

const TweaksPanel = ({ open, onClose, settings, setSettings }) => {
  if (!open) return null;
  const set = (k, v) => setSettings(s => ({ ...s, [k]: v }));
  return (
    <div className="tweaks-panel">
      <div className="tweaks-head">
        <div className="tweaks-title"><IconTweak size={16} /> Tweaks</div>
        <button className="icon-btn" onClick={onClose}><IconX size={14} /></button>
      </div>
      <div className="tweaks-body">
        <TweakGroup label="Theme">
          <SegCtrl options={[{id:'light',label:'Light'},{id:'dark',label:'Dark'}]} value={settings.theme} onChange={v => set('theme', v)} />
        </TweakGroup>
        <TweakGroup label="Accent color">
          <div className="swatch-row">
            {[
              ['#2563eb', 'Blue'],
              ['#7c3aed', 'Violet'],
              ['#0d9488', 'Teal'],
              ['#c4552d', 'Rust'],
              ['#db2777', 'Pink'],
              ['#0f172a', 'Slate'],
            ].map(([c, name]) => (
              <button
                key={c}
                className={`swatch ${settings.accent === c ? 'on' : ''}`}
                style={{ background: c }}
                onClick={() => set('accent', c)}
                title={name}
              />
            ))}
          </div>
        </TweakGroup>
        <TweakGroup label="Information density">
          <SegCtrl options={[{id:'compact',label:'Compact'},{id:'normal',label:'Normal'},{id:'roomy',label:'Roomy'}]} value={settings.density} onChange={v => set('density', v)} />
        </TweakGroup>
        <TweakGroup label="Panel order">
          <SegCtrl options={[{id:'std',label:'Sources · View · Insp.'},{id:'rev',label:'Insp. · View · Sources'}]} value={settings.panelOrder} onChange={v => set('panelOrder', v)} />
        </TweakGroup>
        <TweakGroup label="Object list style">
          <SegCtrl options={[{id:0,label:'Card'},{id:1,label:'Row'},{id:2,label:'Grid'}]} value={settings.objectListStyle} onChange={v => set('objectListStyle', v)} />
        </TweakGroup>
        <TweakGroup label="Overlays (default)">
          <CheckRow label="Bounding boxes" v={settings.showBox} onChange={v => set('showBox', v)} />
          <CheckRow label="Masks" v={settings.showMask} onChange={v => set('showMask', v)} />
          <CheckRow label="Labels" v={settings.showLabel} onChange={v => set('showLabel', v)} />
        </TweakGroup>
      </div>
      <div className="tweaks-foot">
        <span className="tweaks-hint">Changes apply live</span>
      </div>
    </div>
  );
};

const TweakGroup = ({ label, children }) => (
  <div className="tweak-group">
    <div className="tweak-label">{label}</div>
    {children}
  </div>
);

const SegCtrl = ({ options, value, onChange }) => (
  <div className="seg">
    {options.map(o => (
      <button key={o.id} className={`seg-btn ${value === o.id ? 'on' : ''}`} onClick={() => onChange(o.id)}>{o.label}</button>
    ))}
  </div>
);

const CheckRow = ({ label, v, onChange }) => (
  <label className="check-row inline">
    <input type="checkbox" checked={v} onChange={e => onChange(e.target.checked)} />
    <span>{label}</span>
  </label>
);

const ShortcutsModal = ({ open, onClose }) => {
  if (!open) return null;
  const SHORTCUTS = [
    { cat: 'Navigation', items: [
      ['J', 'Next object'],
      ['K', 'Previous object'],
      ['→', 'Next source'],
      ['←', 'Previous source'],
    ]},
    { cat: 'Review actions', items: [
      ['A', 'Approve selected'],
      ['D', 'Delete selected'],
      ['U', 'Reset validation'],
    ]},
    { cat: 'Overlays', items: [
      ['B', 'Toggle bounding boxes'],
      ['M', 'Toggle masks'],
      ['L', 'Toggle labels'],
    ]},
    { cat: 'General', items: [
      ['?', 'Show this help'],
      ['Esc', 'Close panel'],
    ]},
  ];
  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal shortcuts" onClick={e => e.stopPropagation()}>
        <div className="modal-head">
          <div className="modal-title"><IconKeyboard size={16} /> Keyboard shortcuts</div>
          <button className="icon-btn" onClick={onClose}><IconX size={14} /></button>
        </div>
        <div className="shortcuts-grid">
          {SHORTCUTS.map(group => (
            <div key={group.cat} className="shortcut-group">
              <div className="sc-cat">{group.cat}</div>
              {group.items.map(([k, d]) => (
                <div key={k} className="sc-item">
                  <Kbd>{k}</Kbd>
                  <span>{d}</span>
                </div>
              ))}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

Object.assign(window, { ExportScreen, SettingsScreen, TweaksPanel, ShortcutsModal });
