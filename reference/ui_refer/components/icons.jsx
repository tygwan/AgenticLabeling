// Minimal inline SVG icons. Stroke-based, 20px baseline.
const Icon = ({ d, size = 18, stroke = 1.5, fill = 'none', children, style }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill={fill} stroke="currentColor" strokeWidth={stroke} strokeLinecap="round" strokeLinejoin="round" style={style}>
    {d ? <path d={d} /> : children}
  </svg>
);

const IconHome = (p) => <Icon {...p}><path d="M4 11l8-7 8 7v9a1 1 0 0 1-1 1h-4v-6h-6v6H5a1 1 0 0 1-1-1z"/></Icon>;
const IconReview = (p) => <Icon {...p}><rect x="3" y="4" width="18" height="16" rx="2"/><path d="M3 9h18"/><circle cx="8" cy="14" r="2"/><path d="M12 14h6M12 17h4"/></Icon>;
const IconExport = (p) => <Icon {...p}><path d="M12 3v12M7 8l5-5 5 5"/><path d="M5 21h14a2 2 0 0 0 2-2v-3H3v3a2 2 0 0 0 2 2z"/></Icon>;
const IconSettings = (p) => <Icon {...p}><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09a1.65 1.65 0 0 0-1-1.51 1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09a1.65 1.65 0 0 0 1.51-1 1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33h.01a1.65 1.65 0 0 0 1-1.51V3a2 2 0 1 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 1 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/></Icon>;
const IconUpload = (p) => <Icon {...p}><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><path d="M17 8l-5-5-5 5"/><path d="M12 3v12"/></Icon>;
const IconCheck = (p) => <Icon {...p}><path d="M20 6L9 17l-5-5"/></Icon>;
const IconX = (p) => <Icon {...p}><path d="M18 6L6 18M6 6l12 12"/></Icon>;
const IconChevRight = (p) => <Icon {...p}><path d="M9 6l6 6-6 6"/></Icon>;
const IconChevLeft = (p) => <Icon {...p}><path d="M15 6l-6 6 6 6"/></Icon>;
const IconChevDown = (p) => <Icon {...p}><path d="M6 9l6 6 6-6"/></Icon>;
const IconSearch = (p) => <Icon {...p}><circle cx="11" cy="11" r="7"/><path d="M21 21l-4.3-4.3"/></Icon>;
const IconFilter = (p) => <Icon {...p}><path d="M3 4h18l-7 9v6l-4 2v-8z"/></Icon>;
const IconBox = (p) => <Icon {...p}><rect x="4" y="4" width="16" height="16" rx="1"/></Icon>;
const IconMask = (p) => <Icon {...p}><path d="M12 2a7 7 0 0 0-7 7c0 3 2 5 2 7v2a2 2 0 0 0 2 2h6a2 2 0 0 0 2-2v-2c0-2 2-4 2-7a7 7 0 0 0-7-7z"/></Icon>;
const IconLabel = (p) => <Icon {...p}><path d="M20.59 13.41l-7.17 7.17a2 2 0 0 1-2.83 0L2 12V2h10l8.59 8.59a2 2 0 0 1 0 2.82z"/><circle cx="7" cy="7" r="1" fill="currentColor"/></Icon>;
const IconPlay = (p) => <Icon {...p}><path d="M5 3l14 9-14 9V3z"/></Icon>;
const IconPause = (p) => <Icon {...p}><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></Icon>;
const IconImage = (p) => <Icon {...p}><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><path d="M21 15l-5-5L5 21"/></Icon>;
const IconLayers = (p) => <Icon {...p}><path d="M12 2l10 6-10 6L2 8z"/><path d="M2 17l10 6 10-6M2 12l10 6 10-6"/></Icon>;
const IconKeyboard = (p) => <Icon {...p}><rect x="2" y="6" width="20" height="12" rx="2"/><path d="M6 10h.01M10 10h.01M14 10h.01M18 10h.01M6 14h12"/></Icon>;
const IconGrid = (p) => <Icon {...p}><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/></Icon>;
const IconEye = (p) => <Icon {...p}><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></Icon>;
const IconEyeOff = (p) => <Icon {...p}><path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19M14.12 14.12a3 3 0 1 1-4.24-4.24"/><path d="M1 1l22 22"/></Icon>;
const IconTweak = (p) => <Icon {...p}><path d="M4 7h13M4 17h9M19 7h1M15 17h5"/><circle cx="18" cy="7" r="2"/><circle cx="14" cy="17" r="2"/></Icon>;
const IconDot = ({ size = 8, color }) => <span style={{ width: size, height: size, borderRadius: 999, background: color, display: 'inline-block' }} />;
const IconWarn = (p) => <Icon {...p}><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><path d="M12 9v4M12 17h.01"/></Icon>;
const IconSpinner = ({ size = 18 }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
    <path d="M21 12a9 9 0 1 1-6.2-8.5" style={{ transformOrigin: 'center', animation: 'spin 1s linear infinite' }}/>
  </svg>
);
const IconArrowRight = (p) => <Icon {...p}><path d="M5 12h14M13 5l7 7-7 7"/></Icon>;
const IconTrash = (p) => <Icon {...p}><path d="M3 6h18M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/></Icon>;
const IconFlag = (p) => <Icon {...p}><path d="M4 15s1-1 4-1 5 2 8 2 4-1 4-1V3s-1 1-4 1-5-2-8-2-4 1-4 1z"/><path d="M4 22V15"/></Icon>;
const IconDatabase = (p) => <Icon {...p}><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/></Icon>;

Object.assign(window, {
  IconHome, IconReview, IconExport, IconSettings, IconUpload, IconCheck, IconX,
  IconChevRight, IconChevLeft, IconChevDown, IconSearch, IconFilter, IconBox, IconMask,
  IconLabel, IconPlay, IconPause, IconImage, IconLayers, IconKeyboard, IconGrid,
  IconEye, IconEyeOff, IconTweak, IconDot, IconWarn, IconSpinner, IconArrowRight,
  IconTrash, IconFlag, IconDatabase
});
