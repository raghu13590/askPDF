import React, { useState, useEffect, useRef, useMemo } from 'react';
import { Document, Page, pdfjs } from 'react-pdf';
import { Box, Typography } from '@mui/material';
import 'react-pdf/dist/Page/AnnotationLayer.css';
import 'react-pdf/dist/Page/TextLayer.css';

// Set worker source
pdfjs.GlobalWorkerOptions.workerSrc = `//unpkg.com/pdfjs-dist@${pdfjs.version}/build/pdf.worker.min.mjs`;

type BBox = {
    page: number;
    x: number;
    y: number;
    width: number;
    height: number;
    page_height: number;
    page_width: number;
};

type Sentence = {
    id: number;
    text: string;
    bboxes: BBox[];
};

type Props = {
    pdfUrl: string;
    sentences: Sentence[];
    currentId: number | null;
    onJump: (id: number) => void;
    autoScroll: boolean;
    isResizing?: boolean;
    highlightEnabled?: boolean;
};

const PdfViewer = React.memo(function PdfViewer({ pdfUrl, sentences, currentId, onJump, autoScroll, isResizing, highlightEnabled = true }: Props) {
    const [numPages, setNumPages] = useState<number | null>(null);
    const [pageWidth, setPageWidth] = useState<number>(600); // Actual width for PDF rendering
    const [scale, setScale] = useState<number>(1); // CSS scale for instant feedback
    const containerRef = useRef<HTMLDivElement>(null);
    const pdfContentRef = useRef<HTMLDivElement>(null); // For scaling
    const sentenceRefs = useRef<{ [key: number]: HTMLDivElement | null }>({});

    function onDocumentLoadSuccess({ numPages }: { numPages: number }) {
        setNumPages(numPages);
    }

    // Only update pageWidth when resizing ends, use CSS scale for visual feedback during resize
    useEffect(() => {
        if (!containerRef.current) return;

        if (!isResizing) {
            // On resize end, update pageWidth to match container
                    const width = containerRef.current.offsetWidth;
                    setPageWidth(width);
                    setScale(1);
        } else {
            // During resize, use CSS scale for visual feedback
            if (containerRef.current) {
                const width = containerRef.current.offsetWidth;
                if (pageWidth > 0) {
                    setScale(width / pageWidth);
                }
            }
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [isResizing]);

            useEffect(() => {
                if (!containerRef.current || typeof ResizeObserver === 'undefined') return;
                const observer = new ResizeObserver((entries) => {
                    for (const entry of entries) {
                        if (!isResizing && entry.contentRect) {
                            setPageWidth(entry.contentRect.width);
                            setScale(1);
                        }
                    }
                });
                observer.observe(containerRef.current);
                return () => observer.disconnect();
            }, [isResizing]);

    // Pre-calculate sentences by page for performance
    const sentencesByPage = useMemo(() => {
        const map: { [key: number]: (Sentence & { pageBBoxes: BBox[] })[] } = {};
        if (!sentences) return map;
        sentences.forEach(s => {
            if (!s.bboxes) return;
            s.bboxes.forEach(b => {
                if (!map[b.page]) map[b.page] = [];
                let entry = map[b.page].find(e => e.id === s.id);
                if (!entry) {
                    entry = { ...s, pageBBoxes: [] };
                    map[b.page].push(entry);
                }
                entry.pageBBoxes.push(b);
            });
        });
        return map;
    }, [sentences]);

    // Auto-scroll to active sentence
    useEffect(() => {
        if (autoScroll && currentId !== null && sentenceRefs.current[currentId]) {
            sentenceRefs.current[currentId]?.scrollIntoView({
                behavior: 'smooth',
                block: 'center',
            });
        }
    }, [currentId, autoScroll]);

    // Optimized overlay rendering using pre-calculated map
    const getPageOverlays = (pageNumber: number) => {
        if (!highlightEnabled) return null;
        const pageData = sentencesByPage[pageNumber] || [];
        // ONLY render the highlight for the current active sentence
        const activeSentence = pageData.find(s => s.id === currentId);
        if (!activeSentence) return null;

        return (
            <div
                style={{
                    position: 'absolute',
                    top: 0,
                    left: 0,
                    width: '100%',
                    height: '100%',
                    pointerEvents: 'none',
                    zIndex: 5, // Keep it above canvas but potentially below text layer if needed
                }}
            >
                {activeSentence.pageBBoxes.map((bbox, idx) => (
                    <div
                        key={idx}
                        ref={el => {
                            if (idx === 0) {
                                sentenceRefs.current[activeSentence.id] = el;
                            }
                        }}
                        style={{
                            position: 'absolute',
                            left: `${(bbox.x / bbox.page_width) * 100}%`,
                            top: `${((bbox.page_height - (bbox.y + bbox.height)) / bbox.page_height) * 100}%`,
                            width: `${(bbox.width / bbox.page_width) * 100}%`,
                            height: `${(bbox.height / bbox.page_height) * 100}%`,
                            backgroundColor: 'rgba(255, 255, 0, 0.3)',
                        }}
                    />
                ))}
            </div>
        );
    };

    const handlePageDoubleClick = (e: React.MouseEvent, pageNumber: number) => {
        const rect = e.currentTarget.getBoundingClientRect();
        const x = (e.clientX - rect.left) / rect.width;
        const y = (e.clientY - rect.top) / rect.height;

        const pageData = sentencesByPage[pageNumber] || [];
        for (const sentence of pageData) {
            for (const bbox of sentence.pageBBoxes) {
                const bLeft = bbox.x / bbox.page_width;
                const bTop = (bbox.page_height - (bbox.y + bbox.height)) / bbox.page_height;
                const bWidth = bbox.width / bbox.page_width;
                const bHeight = bbox.height / bbox.page_height;

                if (x >= bLeft && x <= bLeft + bWidth && y >= bTop && y <= bTop + bHeight) {
                    onJump(sentence.id);
                    return;
                }
            }
        }
    };

    return (
        <Box
            ref={containerRef}
            sx={{
                height: '100%',
                width: '100%',
                overflow: 'auto',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'stretch',
                bgcolor: 'transparent',
                p: 0,
                m: 0,
                boxSizing: 'border-box',
            }}
        >
            <div
                ref={pdfContentRef}
                style={{
                    transform: scale !== 1 ? `scale(${scale})` : undefined,
                    transformOrigin: 'top left',
                    transition: isResizing ? 'none' : 'transform 0.15s',
                    width: '100%',
                    margin: 0,
                    padding: 0,
                    boxSizing: 'border-box',
                }}
            >
                <Document
                    file={pdfUrl}
                    onLoadSuccess={onDocumentLoadSuccess}
                    loading={<Typography sx={{ p: 2 }}>Loading PDF...</Typography>}
                    error={<Typography color="error" sx={{ p: 2 }}>Failed to load PDF.</Typography>}
                >
                    {Array.from(new Array(numPages), (el, index) => (
                        <Box
                            key={`page_${index + 1}`}
                            onDoubleClick={(e) => handlePageDoubleClick(e, index + 1)}
                            onMouseDown={e => {
                                // Prevent text selection on double click
                                if (e.detail === 2) {
                                    e.preventDefault();
                                }
                            }}
                            sx={{
                                position: 'relative',
                                mb: 0,
                                width: '100%',
                                maxWidth: '100%',
                                p: 0,
                                m: 0,
                                boxSizing: 'border-box',
                                // react-pdf Page container styles
                                '& .react-pdf__Page': {
                                    width: '100% !important',
                                    height: 'auto !important',
                                    margin: '0',
                                },
                                // Ensure canvas doesn't interfere with selection
                                '& canvas': {
                                    display: 'block',
                                    pointerEvents: 'none',
                                    userSelect: 'none',
                                },
                                // Target the text layer for styling if needed
                                '& .react-pdf__Page__textLayer': {
                                    zIndex: 10,
                                }
                            }}
                        >
                            <Page
                                pageNumber={index + 1}
                                width={pageWidth}
                                renderAnnotationLayer={true}
                                renderTextLayer={true}
                            />
                            {/* Overlay Layer */}
                            {getPageOverlays(index + 1)}
                        </Box>
                    ))}
                </Document>
            </div>
        </Box>
    );
});

export default PdfViewer;
