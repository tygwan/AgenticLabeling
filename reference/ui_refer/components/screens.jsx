// Export screen + Settings + Tweaks panel + Shortcuts help.

const ExportScreen = () => {
  const [format, setFormat] = React.useState('yolo');
  const [datasetName, setDatasetName] = React.useState('urban_mobility_v3_q2_2026');
  const [splits, setSplits] = React.useState({ train: 80, val: 15, test: 5 });
  const [filterValidated, setFilterValidated] = React.useState(true);
  const [includeEmpty, setIncludeEmpty] = React.useState(false);
  const totalValidated = 1402;
  const approxSize = format === 'yolo' ? '847 MB' : '1.2 GB';

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
          <div className="sum-stat"><div className="sum-v">{totalValidated.toLocaleString()}</div><div className="sum-k">Objects</div></div>
          <div className="sum-stat"><div className="sum-v">892</div><div className="sum-k">Images</div></div>
          <div className="sum-stat"><div className="sum-v">14</div><div className="sum-k">Classes</div></div>
          <div className="sum-stat"><div className="sum-v">{approxSize}</div><div className="sum-k">~Size</div></div>
        </div>

        <div className="file-tree">
          <div className="tree-line"><span className="tree-k">📦</span> {datasetName}.zip</div>
          <div className="tree-line indent"><span className="tree-k">├─</span> data.yaml</div>
          <div className="tree-line indent"><span className="tree-k">├─</span> images/</div>
          <div className="tree-line indent2"><span className="tree-k">│&nbsp;&nbsp;├─</span> train/ <span className="tree-n">({Math.round(892*splits.train/100)})</span></div>
          <div className="tree-line indent2"><span className="tree-k">│&nbsp;&nbsp;├─</span> val/ <span className="tree-n">({Math.round(892*splits.val/100)})</span></div>
          <div className="tree-line indent2"><span className="tree-k">│&nbsp;&nbsp;└─</span> test/ <span className="tree-n">({Math.round(892*splits.test/100)})</span></div>
          <div className="tree-line indent"><span className="tree-k">└─</span> labels/</div>
          <div className="tree-line indent2"><span className="tree-k">&nbsp;&nbsp;&nbsp;├─</span> train/</div>
          <div className="tree-line indent2"><span className="tree-k">&nbsp;&nbsp;&nbsp;├─</span> val/</div>
          <div className="tree-line indent2"><span className="tree-k">&nbsp;&nbsp;&nbsp;└─</span> test/</div>
        </div>

        <div className="class-dist">
          <div className="dist-title">Class distribution</div>
          {[
            ['person', 512], ['car', 387], ['traffic_light', 198], ['bicycle', 142],
            ['truck', 89], ['motorcycle', 48], ['stop_sign', 26],
          ].map(([cat, n]) => (
            <div key={cat} className="dist-row">
              <span className="dist-cat"><IconDot size={8} color={CATEGORY_COLORS[cat] || '#64748b'} /> {cat}</span>
              <div className="dist-bar"><div className="dist-fill" style={{ width: `${(n/512)*100}%`, background: CATEGORY_COLORS[cat] || '#64748b' }} /></div>
              <span className="dist-n">{n}</span>
            </div>
          ))}
        </div>

        <button className="btn primary large">
          <IconExport size={16} /> Generate {format.toUpperCase()} export
        </button>
        <div className="export-hint">
          <IconWarn size={12} /> Last export: urban_mobility_v3_q1_2026.zip · 2 days ago
        </div>
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

const SettingsScreen = () => (
  <div className="settings-grid">
    <div className="settings-nav">
      <div className="settings-nav-title">Settings</div>
      <button className="settings-nav-item active">Models & runtime</button>
      <button className="settings-nav-item">Paths & storage</button>
      <button className="settings-nav-item">Workspace defaults</button>
      <button className="settings-nav-item">API keys</button>
      <button className="settings-nav-item">Team</button>
    </div>
    <div className="settings-body">
      <div className="settings-section">
        <h3>Detection backend</h3>
        <div className="setting-card">
          <div className="sc-left">
            <div className="sc-title">Florence-2 <span className="sc-ver">v1.2.3</span></div>
            <div className="sc-sub">Grounding DINO for zero-shot class prompts</div>
          </div>
          <div className="sc-right">
            <StatusPill status="running" />
            <button className="btn ghost sm">Configure</button>
          </div>
        </div>
        <div className="setting-card">
          <div className="sc-left">
            <div className="sc-title">Confidence threshold</div>
            <div className="sc-sub">Drop detections below this score</div>
          </div>
          <div className="sc-right">
            <input type="range" min="0" max="100" defaultValue="50" className="inline-range" />
            <span className="mono">0.50</span>
          </div>
        </div>
      </div>

      <div className="settings-section">
        <h3>Segmentation backend</h3>
        <div className="setting-card">
          <div className="sc-left">
            <div className="sc-title">SAM2 <span className="sc-ver">v2.1.0</span></div>
            <div className="sc-sub">Prompted with detection bboxes</div>
          </div>
          <div className="sc-right">
            <StatusPill status="running" />
            <button className="btn ghost sm">Configure</button>
          </div>
        </div>
        <div className="setting-card">
          <div className="sc-left">
            <div className="sc-title">Mask output</div>
            <div className="sc-sub">Store per-object PNG in masks/ directory</div>
          </div>
          <div className="sc-right">
            <select className="select"><option>PNG (binary)</option><option>RLE (compressed)</option><option>Polygon</option></select>
          </div>
        </div>
      </div>

      <div className="settings-section">
        <h3>Registry</h3>
        <div className="setting-card">
          <div className="sc-left">
            <div className="sc-title"><IconDatabase size={14} />&nbsp;SQLite at <span className="mono">./data/registry.db</span></div>
            <div className="sc-sub">1,852 sources · 12,847 objects · 44 categories</div>
          </div>
          <div className="sc-right">
            <button className="btn ghost sm">Open path</button>
            <button className="btn ghost sm">Backup</button>
          </div>
        </div>
      </div>
    </div>
  </div>
);

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
