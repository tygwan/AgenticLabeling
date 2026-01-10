import React, { useState, useRef, useEffect, useCallback } from 'react';
import SplineBackground from './components/SplineBackground';
import Header from './components/Header';
import { AgentAnalysis, AppState, UploadedMedia } from './types';
import { analyzeMedia, fileToBase64, extractVideoFrame } from './services/geminiService';
import { BarChart, Bar, XAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';

// Toast notification component
const Toast: React.FC<{ message: string; type: 'success' | 'error'; onClose: () => void }> = ({ message, type, onClose }) => {
  useEffect(() => {
    const timer = setTimeout(onClose, 3000);
    return () => clearTimeout(timer);
  }, [onClose]);

  return (
    <div
      role="alert"
      aria-live="polite"
      className={`fixed bottom-6 right-6 z-50 px-6 py-4 rounded-xl shadow-2xl backdrop-blur-md flex items-center gap-3 animate-in slide-in-from-bottom-5 fade-in duration-300 ${
        type === 'success' ? 'bg-green-500/90 text-white' : 'bg-red-500/90 text-white'
      }`}
    >
      {type === 'success' ? (
        <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
        </svg>
      ) : (
        <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
        </svg>
      )}
      <span className="font-medium">{message}</span>
      <button
        onClick={onClose}
        className="ml-2 hover:opacity-70 transition-opacity"
        aria-label="Close notification"
      >
        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
        </svg>
      </button>
    </div>
  );
};

const App: React.FC = () => {
  const [appState, setAppState] = useState<AppState>(AppState.IDLE);
  const [media, setMedia] = useState<UploadedMedia | null>(null);
  const [analysis, setAnalysis] = useState<AgentAnalysis | null>(null);
  const [errorMessage, setErrorMessage] = useState<string>('');
  const [isDragging, setIsDragging] = useState(false);
  const [toast, setToast] = useState<{ message: string; type: 'success' | 'error' } | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const abortControllerRef = useRef<AbortController | null>(null);
  const currentRequestIdRef = useRef<number>(0);
  const mediaUrlRef = useRef<string | null>(null);

  // Validate and process file
  const processFile = useCallback((file: File) => {
    const validImageTypes = ['image/jpeg', 'image/png', 'image/gif', 'image/webp'];
    const validVideoTypes = ['video/mp4', 'video/webm', 'video/quicktime'];
    const maxSize = 50 * 1024 * 1024; // 50MB

    if (![...validImageTypes, ...validVideoTypes].includes(file.type)) {
      setToast({ message: 'Unsupported file type. Use JPG, PNG, MP4, or WebM.', type: 'error' });
      return;
    }

    if (file.size > maxSize) {
      setToast({ message: 'File too large. Maximum size is 50MB.', type: 'error' });
      return;
    }

    // Abort any running analysis before switching files
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }

    // Cleanup previous media URL to prevent memory leaks
    if (media?.url) {
      URL.revokeObjectURL(media.url);
    }

    const isVideo = file.type.startsWith('video/');
    const objectUrl = URL.createObjectURL(file);

    // Track URL in ref for cleanup on unmount
    mediaUrlRef.current = objectUrl;

    setMedia({
      url: objectUrl,
      type: isVideo ? 'video' : 'image',
      file: file
    });
    setAppState(AppState.IDLE);
    setErrorMessage('');
    setAnalysis(null);
  }, [media?.url]);

  const handleFileSelect = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    processFile(file);
  };

  // Drag and drop handlers
  const handleDragEnter = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    // Only set to false if leaving the drop zone entirely
    if (e.currentTarget.contains(e.relatedTarget as Node)) return;
    setIsDragging(false);
  }, []);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    const file = e.dataTransfer.files?.[0];
    if (file) {
      processFile(file);
    }
  }, [processFile]);

  const startAnalysis = async () => {
    if (!media) return;

    // Abort any previous analysis
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    abortControllerRef.current = new AbortController();

    // Increment and capture request ID to detect stale responses
    currentRequestIdRef.current += 1;
    const thisRequestId = currentRequestIdRef.current;

    setAppState(AppState.ANALYZING);
    setErrorMessage('');

    try {
      let base64Data = '';
      let mimeType = '';

      if (media.type === 'video') {
        const result = await extractVideoFrame(media.file);
        base64Data = result.base64;
        mimeType = 'image/jpeg';
      } else {
        base64Data = await fileToBase64(media.file);
        mimeType = media.file.type;
      }

      const result = await analyzeMedia(base64Data, mimeType);

      // Check if this request is still the latest (not superseded by a newer one)
      if (thisRequestId !== currentRequestIdRef.current) return;
      if (abortControllerRef.current?.signal.aborted) return;

      setAnalysis(result);
      setAppState(AppState.COMPLETE);
    } catch (error) {
      // Ignore if this request was superseded
      if (thisRequestId !== currentRequestIdRef.current) return;
      if (abortControllerRef.current?.signal.aborted) return;

      console.error(error);
      const message = error instanceof Error ? error.message : 'Analysis failed. Please try again.';
      setErrorMessage(message);
      setAppState(AppState.ERROR);
    }
  };

  // Retry keeps the media, only resets the analysis state
  const retryAnalysis = () => {
    setErrorMessage('');
    startAnalysis();
  };

  const reset = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    if (media?.url) {
      URL.revokeObjectURL(media.url);
    }
    setAppState(AppState.IDLE);
    setMedia(null);
    setAnalysis(null);
    setErrorMessage('');
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  // Export analysis as JSON
  const exportJSON = useCallback(() => {
    if (!analysis) return;

    try {
      const blob = new Blob([JSON.stringify(analysis, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `analysis-${Date.now()}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      setToast({ message: 'JSON exported successfully!', type: 'success' });
    } catch (error) {
      setToast({ message: 'Failed to export JSON', type: 'error' });
    }
  }, [analysis]);

  // Copy analysis to clipboard
  const copyToClipboard = useCallback(async () => {
    if (!analysis) return;

    try {
      await navigator.clipboard.writeText(JSON.stringify(analysis, null, 2));
      setToast({ message: 'Copied to clipboard!', type: 'success' });
    } catch (error) {
      setToast({ message: 'Failed to copy to clipboard', type: 'error' });
    }
  }, [analysis]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      // Use ref to get current URL since state may be stale in cleanup
      if (mediaUrlRef.current) {
        URL.revokeObjectURL(mediaUrlRef.current);
      }
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, []);

  return (
    <div className="relative min-h-screen font-sans text-white selection:bg-brand-accent selection:text-white">
      <SplineBackground />
      <Header />

      <main className="relative z-10 pt-32 px-6 pb-20 max-w-7xl mx-auto flex flex-col items-center">
        
        {/* Hero Section */}
        <div className="text-center max-w-4xl mb-16 space-y-6">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-white/5 border border-white/10 backdrop-blur-sm mb-4">
                <span className="w-2 h-2 rounded-full bg-brand-accent animate-pulse"></span>
                <span className="text-xs font-medium tracking-wide uppercase text-gray-300">Powered by Gemini 2.5 Flash</span>
            </div>
            <h1 className="text-5xl md:text-7xl font-extrabold tracking-tight leading-tight">
                <span className="block hero-text-gradient">Agentic Vision</span>
                <span className="block text-transparent bg-clip-text bg-gradient-to-r from-brand-accent via-brand-pink to-brand-orange">
                   Labeling Studio
                </span>
            </h1>
            <p className="text-lg md:text-xl text-gray-400 max-w-2xl mx-auto leading-relaxed">
                Automate your dataset curation with next-gen AI. Upload images or videos to get instant, structured object detection and semantic context analysis.
            </p>
        </div>

        {/* Interactive Workspace */}
        <div className="w-full grid grid-cols-1 lg:grid-cols-12 gap-8 min-h-[600px]">
            
            {/* Left Panel: Media Input */}
            <div className="lg:col-span-7 flex flex-col gap-6">
                <div
                    className={`glass-panel rounded-3xl p-2 h-full min-h-[500px] flex flex-col relative overflow-hidden group transition-all duration-200 ${
                        isDragging ? 'ring-2 ring-brand-accent scale-[1.02]' : ''
                    }`}
                    onDragEnter={handleDragEnter}
                    onDragLeave={handleDragLeave}
                    onDragOver={handleDragOver}
                    onDrop={handleDrop}
                >
                    {/* Dashed Border Area */}
                    <div className={`absolute inset-2 border-2 border-dashed rounded-2xl pointer-events-none transition-colors ${
                        isDragging ? 'border-brand-accent bg-brand-accent/5' : 'border-white/10 group-hover:border-brand-accent/30'
                    }`}></div>

                    {!media ? (
                        <div
                            className="flex-1 flex flex-col items-center justify-center p-12 text-center cursor-pointer z-10"
                            onClick={() => fileInputRef.current?.click()}
                            onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && (e.preventDefault(), fileInputRef.current?.click())}
                            tabIndex={0}
                            role="button"
                            aria-label="Upload media file. Drag and drop or click to browse."
                        >
                            <div className={`w-20 h-20 rounded-full bg-white/5 flex items-center justify-center mb-6 transition-transform duration-300 ${
                                isDragging ? 'scale-125 bg-brand-accent/20' : 'group-hover:scale-110'
                            }`}>
                                <svg xmlns="http://www.w3.org/2000/svg" className="h-10 w-10 text-brand-accent" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                                </svg>
                            </div>
                            <h3 className="text-2xl font-bold mb-2">
                                {isDragging ? 'Drop to upload!' : 'Drop your media here'}
                            </h3>
                            <p className="text-gray-400 mb-8">Supports JPG, PNG, MP4, WebM (max 50MB)</p>
                            <button
                                className="px-8 py-3 rounded-xl bg-white text-black font-bold hover:bg-gray-200 transition-all transform hover:-translate-y-1 shadow-lg shadow-white/10 focus:outline-none focus:ring-2 focus:ring-brand-accent focus:ring-offset-2 focus:ring-offset-black"
                                aria-label="Browse files to upload"
                            >
                                Browse Files
                            </button>
                        </div>
                    ) : (
                        <div className="relative flex-1 bg-black/50 rounded-2xl overflow-hidden flex items-center justify-center">
                            {media.type === 'video' ? (
                                <video
                                    src={media.url}
                                    controls
                                    className="max-w-full max-h-full object-contain"
                                    aria-label="Uploaded video preview"
                                />
                            ) : (
                                <img
                                    src={media.url}
                                    alt="Uploaded media for analysis"
                                    className="max-w-full max-h-full object-contain"
                                />
                            )}

                            {appState === AppState.IDLE && (
                                <div className="absolute bottom-8 left-1/2 -translate-x-1/2 z-20">
                                    <button
                                        onClick={startAnalysis}
                                        className="px-8 py-4 rounded-full bg-gradient-to-r from-brand-accent to-brand-pink text-white font-bold text-lg shadow-xl shadow-brand-accent/20 hover:scale-105 transition-transform flex items-center gap-3 focus:outline-none focus:ring-2 focus:ring-white focus:ring-offset-2 focus:ring-offset-black"
                                        aria-label="Start AI analysis of uploaded media"
                                    >
                                        <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                                        </svg>
                                        Run Agentic Analysis
                                    </button>
                                </div>
                            )}

                             {appState !== AppState.IDLE && (
                                <button
                                    onClick={reset}
                                    className="absolute top-4 right-4 p-2 rounded-full bg-black/50 hover:bg-white/20 transition-colors backdrop-blur text-white z-20 focus:outline-none focus:ring-2 focus:ring-white"
                                    aria-label="Reset and upload new media"
                                >
                                    <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                                    </svg>
                                </button>
                             )}
                        </div>
                    )}
                    <input
                        type="file"
                        ref={fileInputRef}
                        onChange={handleFileSelect}
                        className="hidden"
                        accept="image/jpeg,image/png,image/gif,image/webp,video/mp4,video/webm,video/quicktime"
                        aria-label="File upload input"
                    />
                </div>
            </div>

            {/* Right Panel: Agent Output */}
            <div className="lg:col-span-5 flex flex-col h-full">
                {appState === AppState.IDLE && !media && (
                     <div className="h-full glass-panel rounded-3xl p-8 flex flex-col items-center justify-center text-center opacity-50">
                        <div className="w-16 h-16 mb-4 rounded-2xl bg-white/5 rotate-12 flex items-center justify-center">
                            <span className="text-3xl">🤖</span>
                        </div>
                        <h4 className="text-xl font-bold mb-2">Agent Idle</h4>
                        <p className="text-sm text-gray-400">Upload media to wake up the Vision Agent.</p>
                     </div>
                )}

                {appState === AppState.ANALYZING && (
                    <div
                        className="h-full glass-panel rounded-3xl p-8 flex flex-col items-center justify-center relative overflow-hidden"
                        role="status"
                        aria-live="polite"
                        aria-busy="true"
                    >
                        <div className="absolute inset-0 bg-gradient-to-b from-transparent to-brand-accent/5 animate-pulse"></div>
                         <div className="relative z-10 text-center">
                             <div className="inline-block mb-6" aria-hidden="true">
                                 <div className="flex gap-2">
                                     <div className="w-3 h-3 bg-brand-accent rounded-full animate-bounce" style={{animationDelay: '0s'}}></div>
                                     <div className="w-3 h-3 bg-brand-pink rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
                                     <div className="w-3 h-3 bg-brand-orange rounded-full animate-bounce" style={{animationDelay: '0.4s'}}></div>
                                 </div>
                             </div>
                             <h3 className="text-2xl font-bold mb-2">Analyzing Scene...</h3>
                             <p className="text-gray-400">Detecting objects, assessing lighting, and generating labels.</p>
                         </div>
                    </div>
                )}

                {/* Error State */}
                {appState === AppState.ERROR && (
                    <div
                        className="h-full glass-panel rounded-3xl p-8 flex flex-col items-center justify-center text-center animate-in fade-in duration-300"
                        role="alert"
                        aria-live="assertive"
                    >
                        <div className="w-16 h-16 mb-6 rounded-full bg-red-500/10 flex items-center justify-center">
                            <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8 text-red-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                            </svg>
                        </div>
                        <h3 className="text-2xl font-bold mb-2 text-white">Analysis Failed</h3>
                        <p className="text-gray-400 mb-6 max-w-xs">
                            {errorMessage || 'An unexpected error occurred during analysis.'}
                        </p>

                        {/* Troubleshooting tips */}
                        <div className="w-full max-w-sm mb-6 p-4 rounded-xl bg-white/5 border border-white/10 text-left">
                            <p className="text-xs font-bold uppercase tracking-wider text-gray-500 mb-2">Troubleshooting</p>
                            <ul className="text-sm text-gray-400 space-y-1">
                                <li>- Check your API key is valid</li>
                                <li>- Ensure file is not corrupted</li>
                                <li>- Try a smaller file size</li>
                            </ul>
                        </div>

                        <div className="flex gap-3">
                            <button
                                onClick={retryAnalysis}
                                className="px-6 py-3 rounded-xl bg-gradient-to-r from-brand-accent to-brand-pink text-white font-bold hover:scale-105 transition-transform focus:outline-none focus:ring-2 focus:ring-brand-accent focus:ring-offset-2 focus:ring-offset-black"
                                aria-label="Retry analysis with same media"
                            >
                                Retry Analysis
                            </button>
                            <button
                                onClick={reset}
                                className="px-6 py-3 rounded-xl bg-white/10 text-white font-medium hover:bg-white/20 transition-colors focus:outline-none focus:ring-2 focus:ring-white focus:ring-offset-2 focus:ring-offset-black"
                                aria-label="Upload different media"
                            >
                                Upload New
                            </button>
                        </div>
                    </div>
                )}

                {appState === AppState.COMPLETE && analysis && (
                    <div className="h-full glass-panel rounded-3xl overflow-hidden flex flex-col animate-in slide-in-from-bottom-10 fade-in duration-500">
                        {/* Header of Card */}
                        <div className="p-6 border-b border-white/10 bg-white/5">
                            <h3 className="text-lg font-bold flex items-center gap-2">
                                <span className="w-2 h-2 rounded-full bg-green-500 shadow-[0_0_10px_rgba(34,197,94,0.5)]"></span>
                                Analysis Complete
                            </h3>
                        </div>

                        {/* Scrollable Content */}
                        <div className="flex-1 overflow-y-auto p-6 space-y-6 custom-scrollbar">
                            
                            {/* Caption */}
                            <div>
                                <h4 className="text-xs font-bold uppercase tracking-wider text-gray-500 mb-2">Generated Caption</h4>
                                <p className="text-lg leading-relaxed font-light text-white/90">
                                    "{analysis.caption}"
                                </p>
                            </div>

                            {/* Tags */}
                            <div>
                                <h4 className="text-xs font-bold uppercase tracking-wider text-gray-500 mb-3">Semantic Tags</h4>
                                <div className="flex flex-wrap gap-2">
                                    {analysis.tags.map((tag, idx) => (
                                        <span key={idx} className="px-3 py-1 rounded-lg bg-white/5 border border-white/10 text-sm text-gray-300 hover:bg-white/10 hover:border-brand-accent/50 transition-colors cursor-default">
                                            #{tag}
                                        </span>
                                    ))}
                                </div>
                            </div>

                             {/* Chart - Confidence Scores */}
                             <div className="h-48 w-full bg-black/20 rounded-xl p-4 border border-white/5">
                                <h4 className="text-xs font-bold uppercase tracking-wider text-gray-500 mb-4">Object Confidence</h4>
                                <ResponsiveContainer width="100%" height="80%">
                                    <BarChart data={analysis.objects.slice(0, 5)}>
                                        <XAxis dataKey="label" tick={{fill: '#666', fontSize: 10}} axisLine={false} tickLine={false} interval={0} />
                                        <Tooltip 
                                            contentStyle={{backgroundColor: '#111', border: '1px solid #333', borderRadius: '8px'}}
                                            itemStyle={{color: '#fff'}}
                                            cursor={{fill: 'rgba(255,255,255,0.05)'}}
                                        />
                                        <Bar dataKey="confidence" radius={[4, 4, 0, 0]}>
                                            {analysis.objects.slice(0, 5).map((entry, index) => (
                                                <Cell key={`cell-${index}`} fill={index % 2 === 0 ? '#7B2BF9' : '#FF0080'} />
                                            ))}
                                        </Bar>
                                    </BarChart>
                                </ResponsiveContainer>
                             </div>

                             {/* Reasoning & Technical */}
                             <div className="grid grid-cols-2 gap-4">
                                <div className="p-4 rounded-xl bg-white/5 border border-white/5">
                                    <h4 className="text-xs font-bold uppercase tracking-wider text-gray-500 mb-1">Lighting</h4>
                                    <p className="text-sm">{analysis.technicalDetails.lighting}</p>
                                </div>
                                <div className="p-4 rounded-xl bg-white/5 border border-white/5">
                                    <h4 className="text-xs font-bold uppercase tracking-wider text-gray-500 mb-1">Composition</h4>
                                    <p className="text-sm">{analysis.technicalDetails.composition}</p>
                                </div>
                             </div>

                             <div>
                                <h4 className="text-xs font-bold uppercase tracking-wider text-gray-500 mb-2">Agent Reasoning</h4>
                                <div className="p-4 rounded-xl bg-brand-accent/5 border border-brand-accent/20 text-sm text-gray-300 italic">
                                    "{analysis.reasoning}"
                                </div>
                             </div>

                        </div>
                         
                        {/* Footer Action */}
                        <div className="p-4 border-t border-white/10 bg-black/20 backdrop-blur flex gap-2">
                             <button
                                onClick={exportJSON}
                                className="flex-1 py-3 rounded-xl bg-white text-black font-bold text-sm hover:bg-gray-200 transition-colors focus:outline-none focus:ring-2 focus:ring-brand-accent focus:ring-offset-2 focus:ring-offset-black flex items-center justify-center gap-2"
                                aria-label="Export analysis results as JSON file"
                             >
                                <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                                </svg>
                                Export JSON
                             </button>
                             <button
                                onClick={copyToClipboard}
                                className="px-4 rounded-xl bg-white/10 text-white hover:bg-white/20 transition-colors focus:outline-none focus:ring-2 focus:ring-white focus:ring-offset-2 focus:ring-offset-black"
                                aria-label="Copy analysis results to clipboard"
                                title="Copy to clipboard"
                             >
                                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7v8a2 2 0 002 2h6M8 7V5a2 2 0 012-2h4.586a1 1 0 01.707.293l4.414 4.414a1 1 0 01.293.707V15a2 2 0 01-2 2h-2M8 7H6a2 2 0 00-2 2v10a2 2 0 002 2h8a2 2 0 002-2v-2" />
                                </svg>
                             </button>
                        </div>
                    </div>
                )}
            </div>
        </div>

        {/* Feature Grid (Below Fold) */}
        <div className="mt-32 w-full grid grid-cols-1 md:grid-cols-3 gap-8">
            <FeatureCard 
                icon="⚡️"
                title="Real-time Detection"
                desc="Instant object recognition powered by the Gemini 2.5 Flash model for ultra-low latency."
            />
            <FeatureCard 
                icon="🎯"
                title="Agentic Reasoning"
                desc="Goes beyond bounding boxes. Our agent understands context, emotion, and technical quality."
            />
            <FeatureCard 
                icon="📹"
                title="Video & Image"
                desc="Seamless support for both static imagery and video streams for comprehensive dataset building."
            />
        </div>

      </main>

        <footer className="relative z-10 py-12 text-center text-gray-600 text-sm">
            <p>© 2024 Agentic Labeling Studio. Powered by Google Gemini & Spline.</p>
        </footer>

        {/* Toast Notification */}
        {toast && (
            <Toast
                message={toast.message}
                type={toast.type}
                onClose={() => setToast(null)}
            />
        )}
    </div>
  );
};

const FeatureCard: React.FC<{icon: string, title: string, desc: string}> = ({ icon, title, desc }) => (
    <div className="glass-panel p-8 rounded-2xl hover:bg-white/5 transition-colors group">
        <div className="text-4xl mb-4 transform group-hover:scale-110 transition-transform duration-300">{icon}</div>
        <h3 className="text-xl font-bold mb-3 text-white">{title}</h3>
        <p className="text-gray-400 leading-relaxed">{desc}</p>
    </div>
);

export default App;