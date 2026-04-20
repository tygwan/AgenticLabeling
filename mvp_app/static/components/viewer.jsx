// Image viewer with bbox + mask + label overlays — CSS-only fit (no JS measurement).

const ImageViewer = ({ source, objects, selectedId, onSelect, show }) => {
  const [imgOk, setImgOk] = React.useState(true);
  // rect is the img's actual rendered rectangle inside .viewer-canvas,
  // computed after the img loads / on resize. Overlays are placed on a
  // layer matching this rect so percentage positions map onto the visible
  // image content rather than the canvas letter-box.
  const [rect, setRect] = React.useState(null);
  const imgRef = React.useRef(null);

  const measure = React.useCallback(() => {
    const img = imgRef.current;
    if (!img) return;
    const parent = img.parentElement;
    if (!parent) return;
    const imgR = img.getBoundingClientRect();
    const parentR = parent.getBoundingClientRect();
    // getBoundingClientRect on an object-fit:contain img returns the img
    // element's box (which equals parent), not the rendered image area.
    // Compute the actual rendered rect from naturalWidth/Height + parent.
    const nat = { w: img.naturalWidth || source.width, h: img.naturalHeight || source.height };
    if (!nat.w || !nat.h) return;
    const boxW = imgR.width, boxH = imgR.height;
    const scale = Math.min(boxW / nat.w, boxH / nat.h);
    const renderedW = nat.w * scale;
    const renderedH = nat.h * scale;
    const left = (imgR.left - parentR.left) + (boxW - renderedW) / 2;
    const top = (imgR.top - parentR.top) + (boxH - renderedH) / 2;
    setRect({ left, top, width: renderedW, height: renderedH });
  }, [source.width, source.height]);

  React.useEffect(() => {
    measure();
    const onResize = () => measure();
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, [measure, source.url]);

  // Re-measure when image load state flips or objects layout changes.
  React.useEffect(() => { measure(); }, [measure, imgOk]);

  return (
    <div className="viewer">
      <div className="viewer-canvas">
        {imgOk ? (
          <img
            ref={imgRef}
            src={source.url}
            width={source.width}
            height={source.height}
            alt={source.file_name}
            className="viewer-img"
            onLoad={measure}
            onError={() => setImgOk(false)}
            draggable={false}
          />
        ) : (
          <div className="viewer-placeholder">
            <IconImage size={48} />
            <div style={{ marginTop: 8, fontSize: 13, opacity: 0.6 }}>Image unavailable</div>
            <div style={{ fontSize: 11, opacity: 0.4, marginTop: 4 }}>{source.file_name}</div>
          </div>
        )}
        {rect && imgOk && (
          <div
            className="viewer-overlay-layer"
            style={{ left: rect.left, top: rect.top, width: rect.width, height: rect.height }}
          >
            {/* Mask overlays (SVG) */}
            <svg
              className="overlay-svg"
              viewBox="0 0 100 100"
              preserveAspectRatio="none"
              style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', pointerEvents: 'none' }}
            >
              {show.mask && objects.map(o => {
                if (o.validated === 'deleted') return null;
                const [x, y, w, h] = o.bbox;
                const color = CATEGORY_COLORS[o.category] || '#64748b';
                const sel = o.object_id === selectedId;
                return (
                  <MaskBlob
                    key={'m_' + o.object_id}
                    x={x * 100} y={y * 100} w={w * 100} h={h * 100}
                    color={color} selected={sel} seed={o.object_id}
                  />
                );
              })}
            </svg>
            {/* Bbox overlays */}
            {show.box && objects.map(o => {
              if (o.validated === 'deleted') return null;
              const [x, y, w, h] = o.bbox;
              const color = CATEGORY_COLORS[o.category] || '#64748b';
              const sel = o.object_id === selectedId;
              return (
                <div
                  key={'b_' + o.object_id}
                  className={`bbox ${sel ? 'selected' : ''} ${o.validated || ''}`}
                  style={{
                    position: 'absolute',
                    left: `${x * 100}%`,
                    top: `${y * 100}%`,
                    width: `${w * 100}%`,
                    height: `${h * 100}%`,
                    '--bb-color': color,
                  }}
                  onClick={(e) => { e.stopPropagation(); onSelect(o.object_id); }}
                >
                  {show.label && (
                    <div className="bbox-label" style={{ background: color }}>
                      <span className="bbox-label-text">{o.category}</span>
                      <span className="bbox-label-conf">{Math.round(o.confidence * 100)}</span>
                    </div>
                  )}
                  {o.validated === 'approved' && (
                    <div className="bbox-badge ok"><IconCheck size={10} stroke={3} /></div>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>
      {/* Zoom / coord hud */}
      <div className="viewer-hud">
        <span className="hud-item">{source.width} × {source.height}</span>
        <span className="hud-sep" />
        <span className="hud-item">{objects.filter(o => o.validated !== 'deleted').length} obj</span>
        <span className="hud-sep" />
        <span className="hud-item">Fit</span>
      </div>
    </div>
  );
};

// Deterministic blob mask
const MaskBlob = ({ x, y, w, h, color, selected, seed }) => {
  const hash = React.useMemo(() => {
    let h = 0;
    for (let i = 0; i < seed.length; i++) h = ((h << 5) - h) + seed.charCodeAt(i);
    return Math.abs(h);
  }, [seed]);
  const pts = React.useMemo(() => {
    const n = 18;
    const out = [];
    for (let i = 0; i < n; i++) {
      const a = (i / n) * Math.PI * 2;
      const r = 0.35 + ((hash >> (i % 12)) & 0xff) / 255 * 0.12;
      const cx = x + w / 2;
      const cy = y + h / 2;
      const px = cx + Math.cos(a) * w * r;
      const py = cy + Math.sin(a) * h * r;
      out.push(`${px.toFixed(2)},${py.toFixed(2)}`);
    }
    return out.join(' ');
  }, [x, y, w, h, hash]);
  return (
    <polygon
      points={pts}
      fill={color}
      fillOpacity={selected ? 0.45 : 0.22}
      stroke={color}
      strokeOpacity={selected ? 0.8 : 0}
      strokeWidth="0.3"
      vectorEffect="non-scaling-stroke"
    />
  );
};

Object.assign(window, { ImageViewer });
